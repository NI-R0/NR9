# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Walker_3D_Ball domain: free-floating 3D walker with ball-kick task.

Single-phase reward with no gates: all reward components are directly active
from the first step. The agent learns standing, walking, and kicking in parallel
instead of being forced through a sequential curriculum.

Foot design:
  Each foot is a flat box geom with separate heel and toe touch sensors.
  The ``flat_foot`` reward is continuous: when any part of a foot touches
  the ground, the foot's tilt angle (via the rotation-matrix zz component)
  determines the reward — ``cos(tilt)`` yields 1.0 for a flat sole and
  0.0 for a vertical foot.  ``max(right, left)`` means one flat foot
  suffices for the full standing reward, so the agent can later lift the
  other foot for walking.

Early termination:
  The episode terminates immediately when the agent falls (torso too low,
  any non-foot body part touching the ground, or knee on the ground).
  No grace period — termination is active from step 1 to avoid wasting
  training time on failed episodes. A lightweight stuck-detection
  (torso height + torso xy + ball xy + action) terminates episodes where
  the agent stops making progress.

Reward normalisation: all positive rewards in [0, 1], penalties in [-1, 0].
Kick reward is [0, 1] based on the angle between ball velocity and target
direction — 0 when the ball moves away or sideways, 1 when it moves
directly toward the target.

Approach shortcut: when the ball is already rolling (speed > 0.2 m/s),
the approach reward is clamped to 1.0 so the agent focuses on positioning
for the kick rather than walking toward the ball again.

Positive weights sum to 1.0 so a perfect step yields reward = 1.0.
Penalty weights are on top of the 1.0 budget, so the realistic optimum
lands slightly below 1.0.

Weight priority: kick and target (the actual task) are weighted highest,
followed by locomotion skills in order of dependency (stand → march →
approach).  This gives the agent clear signal that kicking the ball
is the ultimate goal.

Target curriculum: the target zone shrinks after enough successful hits
during evaluation (success counter incremented externally via
``register_success``).
"""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from loguru import logger
import numpy as np


_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.05   # agent decides ~20×/s (set to 0.02 for 50 Hz)
_STUCK_CHECK_TIME = 1   # s: terminate if agent stuck for this long
_STUCK_EPSILON = 1e-3
_STAND_HEIGHT = 1.6  # full upright height; requires fully extended legs for saturation
_BALL_RADIUS = 0.2
_APPROACH_OFFSET = 0.3  # m behind ball (away from target) for approach point
_TARGET_MIN_DIST = 2.0
_TARGET_MAX_DIST = 5.0
_TARGET_SIZE_MAX = 1.0
_TARGET_SIZE_MIN = 0.2
_TARGET_SHRINK = 0.1
_SUCCESS_THRESHOLD = 5
_TARGET_HIT_BONUS = 5.0  # flat bonus added to reward when ball reaches target

# ---------------------------------------------------------------------------
# Early-termination thresholds
# ---------------------------------------------------------------------------
_TERMINATE_HEIGHT = 0.5       # Torso height (m) below which episode ends
_TERMINATE_GRACE_TIME = 0.0   # Grace period after reset; 0.0 means no grace
_TERMINATE_KNEE_HEIGHT = 0.15  # Knee z-position (m) below which knee is "on ground"
# Any single non-foot touch terminates immediately (_TERMINATE_NON_FOOT check
# in should_terminate uses "> 0" → first contact ends the episode).

# ---------------------------------------------------------------------------
# Reward weights (positive sum to 1.0)
# ---------------------------------------------------------------------------
# Every component is normalised to [0, 1] (rewards) or [-1, 0] (penalties).
#
# **Positive weights** sum to exactly 1.0 → a perfect step (all rewards = 1,
# all penalties = 0, ball moving toward target) yields reward = 1.0.
#
# **Negative (penalty) weights** are *on top* of the 1.0 budget.  They
# pull the reward below 1.0 by an amount proportional to the penalty
# magnitude × weight.
# ---------------------------------------------------------------------------

# --- Positive weights (sum = 1.0) ---
_W_FLAT_FOOT = 0.21    # foot sole flatness
_W_STAND = 0.19        # height + upright (foundation for all locomotion)
_W_WEIGHT_SHIFT = 0.10 # COM lateral shift over one foot (stronger signal for weight transfer)
_W_MARCH = 0.07        # knee lift + hip lift of swing leg (foot-z based, not touch)
_W_STANCE = 0.08       # COM on stance line + forward step
_W_APPROACH = 0.08     # walk toward ball / approach point
_W_KICK = 0.12         # ball direction toward target
_W_TARGET = 0.15       # ball in target zone

# Check that positive weights sum to 1.0
assert abs(sum([
    _W_FLAT_FOOT, _W_STAND, _W_WEIGHT_SHIFT, _W_MARCH,
    _W_STANCE, _W_APPROACH, _W_KICK, _W_TARGET,
]) - 1.0) < 1e-9, "Positive reward weights must sum to 1.0"

# --- Penalty weights (on top of the 1.0 budget) ---
_W_FEET_UNDER = 0.12           # feet not under COM (stronger penalty for sideways-leg posture)
_W_HIP_ALIGN = 0.03            # hip yaw/roll deviation
_W_LEG_SPREAD = 0.04           # feet too far apart laterally
_W_SELF_COLLISION = 0.03       # interpenetration of non-adjacent bodies
_W_TORSO_VEL_ALIGN = 0.05      # torso forward vs. linear velocity direction (reduced — torso may rotate freely)
_W_FEET_TORSO_ALIGN = 0.03     # foot forward vs. torso forward (reduced — allows torso yaw freedom)

# --- Alignment & normalisation constants ---
_VELOCITY_THRESHOLD = 0.15            # m/s: below this, skip torso-velocity penalty
_LEG_SPREAD_THRESHOLD = 0.2           # m: feet lateral distance before penalty
_HIP_YAW_MAX = np.radians(45)         # hip yaw angle (rad) at full penalty
_HIP_ROLL_MAX = np.radians(45)        # hip roll angle (rad) at full penalty
_MARCH_KNEE_TARGET = np.radians(60)   # knee flexion (rad) for full march reward
_MARCH_HIP_PITCH_TARGET = np.radians(45)  # hip pitch (rad) for full march lift
_FEET_UNDER_MAX_OFFSET = 0.25  # m: feet-COM xy offset at full penalty (stricter — feet must be under COM)
_FLAT_FOOT_TOUCH_THRESHOLD = 0.3      # tanh(touch-force) threshold for foot contact
_BALL_SPEED_SATURATE = 3.0            # m/s: ball speed at which kick reward saturates

# March alternation: trapezoidal hold-time curve (in seconds, converted to steps at runtime).
# Ein einzelner Schritt (Fuß anheben → absetzen) dauert ~0.3-0.7s.
#   0 – _MARCH_RAMP_UP       : linear ramp 0 → 1  (0.2s)
#   _MARCH_RAMP_UP – _MARCH_DECAY_START : full reward 1.0  (0.2s plateau)
#   _MARCH_DECAY_START – _MARCH_DECAY_END : linear decay 1 → 0  (1.4s)
# Gesamtdauer bis decay=0: ~1.8s. Verhindert „Stair-Stepping" (gleiche
# Bein wiederholt), ohne den Agent zu zwingen, unnatürlich schnell zu hinken.
_MARCH_RAMP_UP_TIME = 0.2     # s: ramp-up to full march reward
_MARCH_DECAY_START_TIME = 0.4  # s: start decaying march reward
_MARCH_DECAY_END_TIME = 2.0    # s: march reward = 0 (no foot switch)

def _steps(t: float) -> int:
    """Convert a time constant (seconds) to steps based on _CONTROL_TIMESTEP."""
    return max(1, int(t / _CONTROL_TIMESTEP))


# Precomputed step counts (derived from time constants above).
# Change _CONTROL_TIMESTEP freely — these auto-scale.
_MARCH_RAMP_UP = _steps(_MARCH_RAMP_UP_TIME)
_MARCH_DECAY_START = _steps(_MARCH_DECAY_START_TIME)
_MARCH_DECAY_END = _steps(_MARCH_DECAY_END_TIME)
_STUCK_CHECK_STEPS = _steps(_STUCK_CHECK_TIME)
_TERMINATE_GRACE_STEPS = _steps(_TERMINATE_GRACE_TIME)

# Touch sensor names for feet reward
_NON_FOOT_TOUCHES = (
    "torso_touch",
    "right_thigh_touch",
    "left_thigh_touch",
    "right_leg_touch",
    "left_leg_touch",
)
# Foot is split into heel and toe touch sensors (4 sensors total).
# A flat foot = both heel AND toe in contact with the ground.
_FOOT_TOUCHES = (
    "right_heel_touch",
    "right_toe_touch",
    "left_heel_touch",
    "left_toe_touch",
)
_ALL_TOUCHES = _NON_FOOT_TOUCHES + _FOOT_TOUCHES

SUITE = containers.TaggedTasks()
FILE = "walker_3D_ball.xml"


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    xml_path = os.path.join(os.path.dirname(__file__), FILE)
    with open(xml_path, "r") as f:
        xml_string = f.read()
    assets = {f"./common/{k}": v for k, v in common.ASSETS.items()}
    return xml_string, assets


def _make_task(time_limit, random, environment_kwargs):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Walker3DBall(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")
def kick(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Kick task: walk to the ball, kick it into the target."""
    return _make_task(time_limit, random, environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker_3D_Ball domain.

    All name→index lookups are precomputed once in ``_ensure_indices`` and
    subsequent access uses direct NumPy array indexing for speed.  The
    return values of every method are identical to the original named-access
    version.
    """

    # ------------------------------------------------------------------
    # Index precomputation (called once, lazily)
    # ------------------------------------------------------------------

    def _ensure_indices(self):
        """Precompute MuJoCo name→id indices for fast array access.

        Idempotent: computes only on the first call, then sets
        ``self._indices_ready``.
        """
        if getattr(self, "_indices_ready", False):
            return
        model = self.model.ptr
        mjt = mujoco.mjtObj

        # Body IDs
        self._bid_torso = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "torso")
        self._bid_right_foot = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "right_foot")
        self._bid_left_foot = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "left_foot")
        self._bid_target = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "target")
        self._bid_ball = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "ball")
        # Additional body IDs for self-collision check
        self._bid_right_thigh = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "right_thigh")
        self._bid_left_thigh = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "left_thigh")
        self._bid_right_leg = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "right_leg")
        self._bid_left_leg = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "left_leg")
        # Adjacent body pairs (directly connected by joints — exempt from collision penalty)
        self._adjacent_pairs = frozenset([
            frozenset((self._bid_torso, self._bid_right_thigh)),
            frozenset((self._bid_torso, self._bid_left_thigh)),
            frozenset((self._bid_right_thigh, self._bid_right_leg)),
            frozenset((self._bid_left_thigh, self._bid_left_leg)),
            frozenset((self._bid_right_leg, self._bid_right_foot)),
            frozenset((self._bid_left_leg, self._bid_left_foot)),
        ])

        # Joint qpos / qvel addresses
        joint_names = [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle_pitch", "left_ankle_roll",
        ]
        self._qpos_joints = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, n)]
            for n in joint_names
        ], dtype=np.int64)
        self._qpos_root = 0  # root qpos starts at 0
        self._qpos_ball = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "ball_joint")
        ]
        self._qvel_ball = model.jnt_dofadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "ball_joint")
        ]

        # Specific joint qpos addresses for hip_yaw, hip_roll, knee, hip_pitch
        self._qpos_r_hip_yaw = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_hip_yaw")]
        self._qpos_l_hip_yaw = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_hip_yaw")]
        self._qpos_r_hip_roll = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_hip_roll")]
        self._qpos_l_hip_roll = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_hip_roll")]
        self._qpos_r_knee = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_knee")]
        self._qpos_l_knee = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_knee")]
        self._qpos_r_hip_pitch = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_hip_pitch")]
        self._qpos_l_hip_pitch = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_hip_pitch")]

        # Sensor addresses
        self._sensor_linvel = model.sensor_adr[
            mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, "torso_subtreelinvel")]
        # Foot touch sensors: heel and toe for each foot (4 sensors).
        # Order: right_heel, right_toe, left_heel, left_toe
        self._sensor_foot = np.array([
            model.sensor_adr[
                mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, name)]
            for name in _FOOT_TOUCHES
        ], dtype=np.int64)
        # Convenience indices into the 4-element _sensor_foot array
        self._si_r_heel = 0
        self._si_r_toe = 1
        self._si_l_heel = 2
        self._si_l_toe = 3
        # Non-foot touch sensors (torso, right_thigh, left_thigh, right_leg, left_leg)
        self._sensor_non_foot = np.array([
            model.sensor_adr[
                mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, name)]
            for name in _NON_FOOT_TOUCHES
        ], dtype=np.int64)
        # All touch sensors in _ALL_TOUCHES order
        self._sensor_all_touch = np.array([
            model.sensor_adr[
                mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, name)]
            for name in _ALL_TOUCHES
        ], dtype=np.int64)

        # Site IDs for heel/toe touch sites (used by foot_contact_points
        # to get actual contact-point world positions, not body origins).
        # Order matches _FOOT_TOUCHES: right_heel, right_toe, left_heel, left_toe
        self._sid_foot = np.array([
            mujoco.mj_name2id(model, mjt.mjOBJ_SITE, name)
            for name in _FOOT_TOUCHES
        ], dtype=np.int64)

        # Mocap ID for target body
        self._mocap_target = model.body_mocapid[self._bid_target]

        # Geom ID for target_zone
        self._geom_target_zone = mujoco.mj_name2id(
            model, mjt.mjOBJ_GEOM, "target_zone")

        self._indices_ready = True

    # ------------------------------------------------------------------
    # Optimized accessor methods (same return values as named-access versions)
    # ------------------------------------------------------------------

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        self._ensure_indices()
        return self.data.xmat[self._bid_torso, 8]  # zz component

    def torso_height(self):
        """Returns the world-z height of the top end of the torso capsule."""
        self._ensure_indices()
        return self.data.xpos[self._bid_torso, 2] + 0.3 * self.data.xmat[self._bid_torso, 8]

    def torso_xy(self):
        """Returns the [x, y] position of the torso."""
        self._ensure_indices()
        return self.data.xpos[self._bid_torso, :2].copy()

    def horizontal_velocity(self):
        """Returns the horizontal speed of the center-of-mass (xy-plane)."""
        self._ensure_indices()
        linvel = self.data.sensordata[self._sensor_linvel:self._sensor_linvel + 3]
        return float(np.linalg.norm(linvel[:2]))

    def orientations(self):
        """Returns 3D orientations of all bodies (first two rows of xmat).

        Returns columns [xx, xy, xz, yx, yy, yz] for every non-world body.
        The third row is implied by orthogonality, so 6 components suffice.
        This gives the agent full information about forward, lateral, and
        roll/pitch orientation — essential for flat-foot and stance rewards.
        """
        self._ensure_indices()
        return self.data.xmat[1:, :6].ravel()

    def touch_forces(self):
        """Returns touch forces of all body parts (including feet) as a 1-D array.

        Each force is passed through ``tanh(force)`` so that light grazes
        produce ~0 while firm contacts saturate to ~1.  Consistent with
        ``feet_touch`` and ``non_foot_touch`` (no ``-3`` offset).
        """
        self._ensure_indices()
        return np.tanh(self.data.sensordata[self._sensor_all_touch])

    def feet_touch(self):
        """Returns summed tanh-saturated touch force for feet only (all 4 sensors)."""
        self._ensure_indices()
        return float(np.sum(np.tanh(self.data.sensordata[self._sensor_foot])))

    def flat_foot_contact(self):
        """Returns the combined flatness [0, 1] of both feet.

        Uses the absolute world orientation of each foot (xmat zz component).
        No touch sensors involved.

        The lower foot (stance, closer to ground) is weighted 2x, the higher
        foot (swing) is weighted 1x: ``(stance * 2 + swing) / 3``.
        """
        self._ensure_indices()

        # World-normal of each foot (xmat row-2 = local z-axis in world coords)
        # zz = cos(pitch) * cos(roll) -> 1.0 when foot sole is parallel to ground
        r_zz = float(np.clip(self.data.xmat[self._bid_right_foot, 8], 0.0, 1.0))
        l_zz = float(np.clip(self.data.xmat[self._bid_left_foot, 8], 0.0, 1.0))

        # Cubic: cos^3(0)=1.0, cos^3(30)=0.65, cos^3(45)=0.35
        r_flat = r_zz ** 3
        l_flat = l_zz ** 3

        # Lower foot is stance (carries weight), higher foot is swing
        r_z = float(self.data.xpos[self._bid_right_foot, 2])
        l_z = float(self.data.xpos[self._bid_left_foot, 2])

        if r_z <= l_z:
            return float(2.0 * r_flat + l_flat) / 3.0
        else:
            return float(r_flat + 2.0 * l_flat) / 3.0


    def foot_contact_points(self):
        """Returns the xy positions of all foot contact points on the ground.

        A contact point is added for each foot-part sensor (heel/toe) whose
        tanh(touch) exceeds ``_FLAT_FOOT_TOUCH_THRESHOLD``.  The xy position
        is taken from the corresponding **site** world position
        (``data.site_xpos``), not the foot body origin, so heel and toe
        of the same foot yield distinct points — critical for a meaningful
        support polygon.
        """
        self._ensure_indices()
        touches = np.tanh(self.data.sensordata[self._sensor_foot])
        in_contact = touches > _FLAT_FOOT_TOUCH_THRESHOLD
        if not np.any(in_contact):
            return np.empty((0, 2))
        # site_xpos has shape (nsite, 3); use precomputed site IDs
        return self.data.site_xpos[self._sid_foot[in_contact], :2].copy()

    def com_ground_projection(self):
        """Returns the [x, y] ground projection of the centre of mass.

        Uses ``subtree_com`` of the torso (the full upper-body COM).
        """
        self._ensure_indices()
        return self.data.subtree_com[self._bid_torso, :2].copy()

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
        """Returns the convex hull of a set of 2D points (gift wrapping / Jarvis march).

        Args:
            points: (N, 2) array of 2D points.

        Returns:
            (M, 2) array of hull vertices in counter-clockwise order,
            where M ≤ N.  For N ≤ 2 the input is returned as-is.
        """
        pts = np.asarray(points)
        n = len(pts)
        if n <= 2:
            return pts.copy()

        # Find the leftmost point (guaranteed to be on the hull).
        start = int(np.argmin(pts[:, 0]))
        hull_indices = []
        p = start
        while True:
            hull_indices.append(p)
            q = (p + 1) % n
            for i in range(n):
                if i == p:
                    continue
                # Cross product: positive ⇒ i is more counter-clockwise than q
                # relative to p.  We want the most counter-clockwise point.
                cross = (
                    (pts[i, 0] - pts[p, 0]) * (pts[q, 1] - pts[p, 1])
                    - (pts[i, 1] - pts[p, 1]) * (pts[q, 0] - pts[p, 0])
                )
                if cross < 0:
                    q = i
            p = q
            if p == start:
                break
        return pts[hull_indices]

    def com_to_support_distance(self):
        """Returns the distance from the COM ground projection to the support polygon.

        The support polygon is the convex hull of all foot contact points.
        Returns 0.0 if the COM is inside the polygon (stable), or the
        positive distance to the nearest polygon edge if outside (unstable).
        Returns a large value (1e6) if no foot contacts exist (airborne).
        """
        contact_pts = self.foot_contact_points()
        if len(contact_pts) < 1:
            return 1e6
        if len(contact_pts) == 1:
            # Single contact point: distance to that point
            return float(np.linalg.norm(
                self.com_ground_projection() - contact_pts[0]))

        com_xy = self.com_ground_projection()

        if len(contact_pts) == 2:
            # Two points: distance to the line segment
            p1, p2 = contact_pts
            seg = p2 - p1
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-12:
                return float(np.linalg.norm(com_xy - p1))
            t = np.clip(np.dot(com_xy - p1, seg) / seg_len_sq, 0.0, 1.0)
            proj = p1 + t * seg
            return float(np.linalg.norm(com_xy - proj))

        # 3+ points: use convex hull (pure NumPy, no scipy needed)
        hull_pts = self._convex_hull_2d(contact_pts)

        # If COM is inside the hull, all signed distances (with outward
        # normals) are ≤ 0, so max(0, max_signed) = 0.  If the COM is
        # outside at least one edge, that edge's signed distance is > 0
        # and the maximum gives the perpendicular distance to the nearest
        # violated edge — a lower bound on the true Euclidean distance,
        # sufficient for reward shaping.
        max_dist = float("-inf")
        n = len(hull_pts)
        centroid = np.mean(hull_pts, axis=0)
        for i in range(n):
            p1 = hull_pts[i]
            p2 = hull_pts[(i + 1) % n]
            edge = p2 - p1
            edge_len = float(np.linalg.norm(edge))
            if edge_len < 1e-12:
                continue
            normal = np.array([edge[1], -edge[0]]) / edge_len
            # Ensure normal points outward (away from centroid)
            if np.dot(normal, centroid - p1) > 0:
                normal = -normal
            dist = float(np.dot(com_xy - p1, normal))
            if dist > max_dist:
                max_dist = dist

        return max(0.0, max_dist)

    def non_foot_touch(self):
        """Returns summed tanh-saturated touch force for non-foot body parts."""
        self._ensure_indices()
        return float(np.sum(np.tanh(self.data.sensordata[self._sensor_non_foot])))

    def feet_horizontal_distance(self):
        """Returns the xy-distance between the two feet."""
        self._ensure_indices()
        right_foot = self.data.xpos[self._bid_right_foot, :2]
        left_foot = self.data.xpos[self._bid_left_foot, :2]
        return float(np.linalg.norm(right_foot - left_foot))

    def feet_lateral_distance(self):
        """Returns the absolute y-distance between the two feet."""
        self._ensure_indices()
        return float(abs(self.data.xpos[self._bid_right_foot, 1] -
                         self.data.xpos[self._bid_left_foot, 1]))

    def ball_position(self):
        """Returns the [x, y, z] position of the ball."""
        self._ensure_indices()
        return self.data.qpos[self._qpos_ball:self._qpos_ball + 3].copy()

    def ball_velocity(self):
        """Returns the [vx, vy, vz, wx, wy, wz] velocity of the ball."""
        self._ensure_indices()
        return self.data.qvel[self._qvel_ball:self._qvel_ball + 6].copy()

    def ball_xy(self):
        """Returns the [x, y] position of the ball."""
        self._ensure_indices()
        return self.data.qpos[self._qpos_ball:self._qpos_ball + 2].copy()

    def ball_linear_velocity_xy(self):
        """Returns the [vx, vy] linear velocity of the ball."""
        self._ensure_indices()
        return self.data.qvel[self._qvel_ball:self._qvel_ball + 2].copy()

    def target_position(self):
        """Returns the [x, y, z] world position of the target mocap body."""
        self._ensure_indices()
        return self.data.xpos[self._bid_target].copy()

    def target_xy(self):
        """Returns the [x, y] world position of the target."""
        self._ensure_indices()
        return self.data.xpos[self._bid_target, :2].copy()

    def set_target_position(self, xy):
        """Moves the mocap target body to ``(x, y, 0.1)`` and recomputes kinematics.

        ``physics.forward()`` is needed because ``data.xpos`` for mocap bodies is
        only updated during kinematics, which runs inside ``physics.step()`` - not
        when ``mocap_pos`` is set directly.
        """
        self._ensure_indices()
        self.data.mocap_pos[self._mocap_target] = [xy[0], xy[1], 0.1]
        self.forward()

    def set_target_size(self, half_size):
        """Sets the target zone geom half-size in xy (box)."""
        self._ensure_indices()
        self.model.ptr.geom_size[self._geom_target_zone] = [half_size, half_size, 0.05]

    def get_target_size(self):
        """Returns the current target zone half-size (xy)."""
        self._ensure_indices()
        return float(self.model.ptr.geom_size[self._geom_target_zone, 0])

    def hip_yaw_angles(self):
        """Returns [right_hip_yaw, left_hip_yaw] joint angles in radians."""
        self._ensure_indices()
        return np.array([
            self.data.qpos[self._qpos_r_hip_yaw],
            self.data.qpos[self._qpos_l_hip_yaw],
        ]).ravel()

    def hip_roll_angles(self):
        """Returns [right_hip_roll, left_hip_roll] joint angles in radians."""
        self._ensure_indices()
        return np.array([
            self.data.qpos[self._qpos_r_hip_roll],
            self.data.qpos[self._qpos_l_hip_roll],
        ]).ravel()

    def foot_heights(self):
        """Returns [right_foot_z, left_foot_z] world-z heights of the feet."""
        self._ensure_indices()
        return np.array([
            self.data.xpos[self._bid_right_foot, 2],
            self.data.xpos[self._bid_left_foot, 2],
        ])

    def knee_heights(self):
        """Returns [right_knee_z, left_knee_z] world-z heights of the knee joints.

        Uses the world position of the right_leg / left_leg body (the knee joint
        sits at the origin of these bodies in the kinematic tree).
        """
        self._ensure_indices()
        return np.array([
            self.data.xpos[self._bid_right_leg, 2],
            self.data.xpos[self._bid_left_leg, 2],
        ])

    def joint_positions(self):
        """Returns all non-root joint angles as a 1-D array (12 joints).

        Order: right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee,
        right_ankle_pitch, right_ankle_roll, left_hip_yaw, left_hip_roll,
        left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll.
        """
        self._ensure_indices()
        return self.data.qpos[self._qpos_joints].copy()

    def knee_angles(self):
        """Returns [right_knee, left_knee] joint angles in radians."""
        self._ensure_indices()
        return np.array([
            self.data.qpos[self._qpos_r_knee],
            self.data.qpos[self._qpos_l_knee],
        ]).ravel()

    def hip_pitch_angles(self):
        """Returns [right_hip_pitch, left_hip_pitch] joint angles in radians."""
        self._ensure_indices()
        return np.array([
            self.data.qpos[self._qpos_r_hip_pitch],
            self.data.qpos[self._qpos_l_hip_pitch],
        ]).ravel()

    def com_lateral_to_foot(self):
        """Returns [com_to_right_foot_y, com_to_left_foot_y] distances.

        A small absolute value means the COM is directly above that foot.
        """
        self._ensure_indices()
        com_y = self.data.subtree_com[self._bid_torso, 1]
        right_foot_y = self.data.xpos[self._bid_right_foot, 1]
        left_foot_y = self.data.xpos[self._bid_left_foot, 1]
        return np.array([com_y - right_foot_y, com_y - left_foot_y])

    def feet_xy_offset(self):
        """Returns the xy-distance from the stance foot to the centre of mass.

        Stance foot = the lower foot (smaller z).  If both feet are at similar
        height (dual support), both are included.  Swing feet in the air are
        naturally ignored because they are higher.

        Returns the offset of ground-feet only.  If no clear ground foot,
        returns 0.0 (no penalty applicable).
        """
        self._ensure_indices()
        com_xy = self.data.subtree_com[self._bid_torso, :2]
        r_z = float(self.data.xpos[self._bid_right_foot, 2])
        l_z = float(self.data.xpos[self._bid_left_foot, 2])

        # Lower foot is stance; include the other only if within 0.05 m (dual support)
        min_z = min(r_z, l_z)
        offsets = []
        if r_z < min_z + 0.05:
            offsets.append(float(np.linalg.norm(com_xy - self.data.xpos[self._bid_right_foot, :2])))
        if l_z < min_z + 0.05:
            offsets.append(float(np.linalg.norm(com_xy - self.data.xpos[self._bid_left_foot, :2])))

        if len(offsets) == 0:
            return 0.0
        return float(np.mean(offsets))

    def self_collision_penalty(self):
        """Returns the total self-collision penalty [-1, 0].

        Iterates over all active MuJoCo contacts.  Contacts between
        body-pairs that are **directly connected by a joint** (adjacent
        in the kinematic tree) are ignored — they are physically allowed
        to overlap.  All other interpenetrations are penalised by their
        contact depth (``contact.dist`` is negative when bodies overlap).

        Returns ``-min(1.0, total_penetration / depth_scale)`` so the
        output is in [-1, 0].  Zero means no self-collision.
        """
        self._ensure_indices()
        data = self.data
        model = self.model.ptr
        depth_scale = 0.05  # 5cm of total penetration = fully saturated penalty

        total_depth = 0.0
        for i in range(int(data.ncon)):
            c = data.contact[i]
            # Find the body that owns each geom of this contact
            geom1_body = model.geom_bodyid[int(c.geom1)]
            geom2_body = model.geom_bodyid[int(c.geom2)]

            # Skip if bodies are the same (self-contact within one body)
            if geom1_body == geom2_body:
                continue

            # Skip adjacent pairs (parent-child in kinematic tree)
            pair = frozenset((geom1_body, geom2_body))
            if pair in self._adjacent_pairs:
                continue

            # Skip contacts involving the ball, floor, target, or worldbody
            if geom1_body == self._bid_ball or geom2_body == self._bid_ball:
                continue
            if geom1_body == self._bid_target or geom2_body == self._bid_target:
                continue
            # body 0 = worldbody (floor etc.)
            if geom1_body == 0 or geom2_body == 0:
                continue

            # dist < 0 means interpenetration; deeper = more negative
            if c.dist < 0:
                total_depth += abs(c.dist)

        return float(-min(1.0, total_depth / depth_scale))

    def torso_forward_xy(self):
        """Returns the xy-component of the torso's forward (local +x) vector."""
        self._ensure_indices()
        return self.data.xmat[self._bid_torso, :2].copy()

    def foot_forward_xy(self, foot_id: int):
        """Returns the xy-component of a foot's forward (local +x) vector."""
        self._ensure_indices()
        return self.data.xmat[foot_id, :2].copy()

    def torso_velocity_align(self):
        """Returns alignment [0, 1] between torso forward and linear velocity.

        Returns 1.0 when torso faces the direction of motion, 0.0 when
        perpendicular or moving backwards. Returns 0.0 when speed is
        below ``_VELOCITY_THRESHOLD`` (penalty not applicable when still).
        """
        self._ensure_indices()
        fwd = self.torso_forward_xy()
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm < 1e-6:
            return 0.0

        vel_xy = self.data.qvel[0:2].copy()
        # Root qvel[0:2] = translational velocity of torso (freejoint: x, y, z, qx, qy, qz)
        speed = float(np.linalg.norm(vel_xy))
        if speed < _VELOCITY_THRESHOLD:
            return 0.0  # no penalty when still

        cos_angle = float(np.dot(fwd / fwd_norm, vel_xy / speed))
        return float(max(0.0, cos_angle))  # [0, 1]

    def feet_torso_align(self):
        """Returns average alignment [0, 1] between each foot's forward and torso forward.

        Only counts feet that are in contact with the ground.  If no foot
        is in contact, returns 0.0 (fully misaligned → penalty applies).
        """
        self._ensure_indices()
        touches = np.tanh(self.data.sensordata[self._sensor_foot])
        # Average heel+toe per foot
        r_contact = (touches[0] + touches[1]) / 2.0 > _FLAT_FOOT_TOUCH_THRESHOLD
        l_contact = (touches[2] + touches[3]) / 2.0 > _FLAT_FOOT_TOUCH_THRESHOLD

        torso_fwd = self.torso_forward_xy()
        tf_norm = float(np.linalg.norm(torso_fwd))
        if tf_norm < 1e-6:
            return 0.0
        torso_fwd = torso_fwd / tf_norm

        align_sum = 0.0
        count = 0
        if r_contact:
            r_fwd = self.foot_forward_xy(self._bid_right_foot)
            rf_norm = float(np.linalg.norm(r_fwd))
            if rf_norm > 1e-6:
                align_sum += max(0.0, np.dot(r_fwd / rf_norm, torso_fwd))
                count += 1
        if l_contact:
            l_fwd = self.foot_forward_xy(self._bid_left_foot)
            lf_norm = float(np.linalg.norm(l_fwd))
            if lf_norm > 1e-6:
                align_sum += max(0.0, np.dot(l_fwd / lf_norm, torso_fwd))
                count += 1

        if count == 0:
            return 0.0  # no feet in contact → penalise
        return float(align_sum / count)


class Walker3DBall(base.Task):
    """3D walker with direct reward (no gates) and target curriculum.

    All reward components are directly active from the first step.
    The agent learns standing, walking, and kicking in parallel.

    Target curriculum: ``register_success`` increments a counter.  After
    ``_SUCCESS_THRESHOLD`` consecutive successes the target zone shrinks
    by ``_TARGET_SHRINK`` (down to ``_TARGET_SIZE_MIN``).  A failure
    resets the consecutive-success counter.
    """

    def __init__(self, random=None):
        self._target_size = _TARGET_SIZE_MAX
        self._consecutive_successes = 0
        self._target_pos = None
        self._reward_components: dict[str, float] = {}
        # March alternation tracking
        self._last_swing_leg: str | None = None
        self._confirmed_swing_leg: str | None = None
        self._same_swing_count: int = 0
        # Stuck detection
        self._step_count = 0
        self._stuck_count = 0
        self._last_state: np.ndarray | None = None
        self._last_action: np.ndarray | None = None
        super().__init__(random=random)

    def register_success(self):
        """Call when the agent successfully hits the target during evaluation.

        After ``_SUCCESS_THRESHOLD`` consecutive successes the target zone
        shrinks.  This is intended to be called from the training/eval loop.
        """
        self._consecutive_successes += 1
        if (
            self._consecutive_successes >= _SUCCESS_THRESHOLD
            and self._target_size > _TARGET_SIZE_MIN
        ):
            self._target_size = max(
                _TARGET_SIZE_MIN, self._target_size - _TARGET_SHRINK
            )
            self._consecutive_successes = 0

    def register_failure(self):
        """Reset the consecutive-success counter."""
        self._consecutive_successes = 0

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Resets the free-floating root to the nominal upright pose, randomizes
        joints, places the ball at its start position, and randomly places the
        target.  Also applies the current curriculum target size.
        """
        self._setup_episode(physics)
        super().initialize_episode(physics)

    def _setup_episode(self, physics):
        """Reset walker pose, ball, target, and per-episode bookkeeping.

        Shared between ``initialize_episode`` (full episode reset) and
        mid-episode respawn after a successful target hit.

        Calls ``physics.forward()`` at the end so that ``xpos``/``xmat`` are
        up-to-date before the next reward evaluation or observation.
        """
        # Spawn: walker at random xy, ball at random distance/angle from walker.
        spawn_x = self.random.uniform(-1.5, 1.5)
        spawn_y = self.random.uniform(-1.5, 1.5)
        spawn_yaw = self.random.uniform(0, 2 * np.pi)  # random torso orientation
        # freejoint qpos: [x, y, z, qw, qx, qy, qz] — yaw → quaternion about z-axis
        physics.named.data.qpos["root"] = [
            spawn_x, spawn_y, 1.6,
            np.cos(spawn_yaw / 2.0), 0.0, 0.0, np.sin(spawn_yaw / 2.0)
        ]
        physics.named.data.qvel["root"] = 0.0
        joints = physics.joint_positions()
        joints += self.random.uniform(-0.175, 0.175, size=joints.shape)
        physics.data.qpos[physics._qpos_joints] = joints

        ball_dist = self.random.uniform(1.0, 2.5)
        ball_angle = self.random.uniform(0, 2 * np.pi)
        ball_x = spawn_x + ball_dist * np.cos(ball_angle)
        ball_y = spawn_y + ball_dist * np.sin(ball_angle)
        physics.named.data.qpos["ball_joint"] = [ball_x, ball_y, _BALL_RADIUS, 1, 0, 0, 0]
        physics.named.data.qvel["ball_joint"] = 0.0

        physics.set_target_size(self._target_size)
        self._place_target(physics)

        # Ensure ball is not spawning inside the target zone
        attempts = 0
        while (np.linalg.norm(physics.ball_xy() - physics.target_xy()) <
               physics.get_target_size() + _BALL_RADIUS + 0.5 and attempts < 20):
            ball_dist = self.random.uniform(1.0, 2.5)
            ball_angle = self.random.uniform(0, 2 * np.pi)
            ball_x = spawn_x + ball_dist * np.cos(ball_angle)
            ball_y = spawn_y + ball_dist * np.sin(ball_angle)
            physics.named.data.qpos["ball_joint"][:2] = [ball_x, ball_y]
            attempts += 1

        self._step_count = 0
        self._stuck_count = 0
        self._last_state = None
        self._last_action = None
        self._last_swing_leg = None
        self._confirmed_swing_leg = None
        self._same_swing_count = 0

        # Recompute kinematics so xpos/xmat reflect the new joint positions
        physics.forward()

    def _place_target(self, physics):
        """Randomly places the target at a random angle and distance."""
        angle = self.random.uniform(0, 2 * np.pi)
        dist = self.random.uniform(_TARGET_MIN_DIST, _TARGET_MAX_DIST)
        self._target_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        physics.set_target_position(self._target_pos)

    def _respawn_after_hit(self, physics):
        """Full mid-episode respawn: walker, ball, and target are reset.

        Called when the ball reaches the target.  The episode continues
        (no termination) — the agent must stand up and approach the new
        ball from scratch.
        """
        self._setup_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body state, ball, target, touches, and joints."""
        obs = collections.OrderedDict()
        obs["orientations"] = physics.orientations()
        obs["height"] = physics.torso_height()
        obs["velocity"] = physics.velocity()
        obs["ball_position"] = physics.ball_position()
        obs["ball_velocity"] = physics.ball_velocity()[:3]
        obs["target_position"] = physics.target_xy()
        obs["touches"] = physics.touch_forces()
        obs["joint_positions"] = physics.joint_positions()
        return obs

    def should_terminate(self, physics) -> bool:
        self._step_count += 1

        physics._ensure_indices()

        # 1. Torso too low
        if physics.torso_height() < _TERMINATE_HEIGHT:
            if getattr(self, "_debug_terminate", False):
                logger.warning(f"Terminate: torso_height={physics.torso_height():.3f} < {_TERMINATE_HEIGHT}")
            return True

        # 2. Any non-foot body part touching ground (first contact → terminate)
        non_foot = physics.non_foot_touch()
        if non_foot > 0:
            if getattr(self, "_debug_terminate", False):
                logger.warning(f"Terminate: non_foot_touch={non_foot:.4f}")
            return True

        # 3. Either knee on ground
        knee_z = physics.knee_heights()
        if knee_z[0] < _TERMINATE_KNEE_HEIGHT or knee_z[1] < _TERMINATE_KNEE_HEIGHT:
            if getattr(self, "_debug_terminate", False):
                logger.warning(f"Terminate: knee_z=({knee_z[0]:.3f}, {knee_z[1]:.3f}) < {_TERMINATE_KNEE_HEIGHT}")
            return True

        # 4. Stuck detection – compare COM xyz + both foot xyz + action.
        # Uses the actual body pose, not the full observation or ball position,
        # so rolling ball drift does not falsely reset the counter.
        com_xy = physics.com_ground_projection()
        com_z = physics.data.subtree_com[physics._bid_torso, 2]
        right_foot_xyz = physics.data.xpos[physics._bid_right_foot]
        left_foot_xyz = physics.data.xpos[physics._bid_left_foot]
        state = np.hstack([com_xy, [com_z], right_foot_xyz, left_foot_xyz])
        current_act = physics.control().copy()

        if self._last_state is not None:
            if (np.max(np.abs(state - self._last_state)) < _STUCK_EPSILON and
                    np.max(np.abs(current_act - self._last_action)) < _STUCK_EPSILON):
                self._stuck_count += 1
            else:
                self._stuck_count = 0
        else:
            self._stuck_count = 0

        self._last_state = state.copy()
        self._last_action = current_act

        if self._stuck_count >= _STUCK_CHECK_STEPS:
            logger.warning(
                f"Episode terminated: agent stuck "
                f"({_STUCK_CHECK_STEPS} steps / {_STUCK_CHECK_TIME}s)"
            )
            return True

        return False

    def get_reward(self, physics) -> float:
        """Reward with 8 positive components (summing to 1.0) + penalties.

        Positive rewards in [0, 1], penalties in [-1, 0].
        All components active from step 1 (no gates / no curriculum phases).
        """
        physics._ensure_indices()

        # ------------------------------------------------------------------
        # Shared quantities
        # ------------------------------------------------------------------
        knee = physics.knee_angles()
        sensor_foot = physics._sensor_foot

        # ------------------------------------------------------------------
        # 1. Stand [0, 1] — height + upright + knee extension
        # ------------------------------------------------------------------
        standing = float(np.clip(physics.torso_height() / _STAND_HEIGHT, 0.0, 1.0))
        upright = float((1.0 + physics.torso_upright()) / 2.0)
        # Knee extension: reward fully straight knees (angle≈0), penalise flexion (angle<0)
        knee_extension = float(np.clip(1.0 + np.mean(knee) / np.radians(30), 0.0, 1.0))
        stand_reward = 0.48 * standing + 0.42 * upright + 0.10 * knee_extension

        # ------------------------------------------------------------------
        # 2. Flat foot [0, 1] — foot sole flatness
        # ------------------------------------------------------------------
        flat_foot_reward = physics.flat_foot_contact()

        # ------------------------------------------------------------------
        # 3. Penalties
        # ------------------------------------------------------------------
        # Hip alignment [-1, 0]: mean normalised deviation of yaw + roll
        hip_yaw = physics.hip_yaw_angles()
        hip_roll = physics.hip_roll_angles()
        hip_align = -float(np.clip(
            (np.mean(np.abs(hip_yaw)) / _HIP_YAW_MAX + np.mean(np.abs(hip_roll)) / _HIP_ROLL_MAX)
            / 2.0,
            0.0, 1.0,
        ))

        # Leg spread [-1, 0]: feet too far apart
        feet_dist = physics.feet_horizontal_distance()
        leg_spread = -float(
            np.clip((feet_dist - _LEG_SPREAD_THRESHOLD) / _LEG_SPREAD_THRESHOLD, 0.0, 1.0)
        )

        # Feet under torso [-1, 0]: feet should be beneath COM
        feet_offset = physics.feet_xy_offset()
        feet_under = float(np.clip(feet_offset / _FEET_UNDER_MAX_OFFSET, 0.0, 1.0))

        # Self-collision [-1, 0]
        self_collision = physics.self_collision_penalty()

        # Torso-velocity alignment [-1, 0]: torso forward vs. motion direction
        # When speed < _VELOCITY_THRESHOLD, returns 0 (no penalty when still).
        # Otherwise 0.0 (perpendicular/backwards) → -1.0, 1.0 (aligned) → 0.0.
        torso_vel_align = physics.torso_velocity_align()

        # Feet-torso alignment [-1, 0]: each ground-contact foot should face torso direction
        feet_torso_align = physics.feet_torso_align()

        # ------------------------------------------------------------------
        # 4. Weight shift [0, 1] — COM laterally over one foot
        # ------------------------------------------------------------------
        com_to_feet = physics.com_lateral_to_foot()
        half_foot_dist = max(physics.feet_lateral_distance() / 2.0, 1e-6)
        shift_right = 1.0 - float(np.clip(np.abs(com_to_feet[0]) / half_foot_dist, 0.0, 1.0))
        shift_left = 1.0 - float(np.clip(np.abs(com_to_feet[1]) / half_foot_dist, 0.0, 1.0))
        weight_shift_reward = float(np.clip(max(shift_right, shift_left), 0.0, 1.0))

        # ------------------------------------------------------------------
        # 5. March [0, 1] — single support + swing leg lift + alternation
        # Swing detection: higher foot (z-position) is the swing leg.
        r_foot_z = float(physics.data.xpos[physics._bid_right_foot, 2])
        l_foot_z = float(physics.data.xpos[physics._bid_left_foot, 2])
        foot_z_diff = abs(r_foot_z - l_foot_z)

        # Single support: one foot clearly higher than the other
        SWING_MIN_LIFT = 0.05  # m: min z-diff to identify swing leg
        if foot_z_diff < SWING_MIN_LIFT:
            single_support = 0.0
            swing = -1
        elif r_foot_z > l_foot_z:
            single_support = 1.0
            swing = 0  # right is higher -> right is swing leg
        else:
            single_support = 1.0
            swing = 1  # left is higher -> left is swing leg

        hip_pitch = physics.hip_pitch_angles()
        if swing >= 0:
            knee_lift = float(np.clip(-knee[swing] / _MARCH_KNEE_TARGET, 0.0, 1.0))
            hip_lift = float(np.clip(hip_pitch[swing] / _MARCH_HIP_PITCH_TARGET, 0.0, 1.0))
            march_lift = (knee_lift + hip_lift) / 2.0
        else:
            march_lift = 0.0

        # Alternation: penalise "stair-stepping" (same leg swinging repeatedly)
        current_swing_leg = ("right", "left")[swing] if swing >= 0 else None
        if current_swing_leg is not None:
            if self._last_swing_leg is None:
                self._confirmed_swing_leg = current_swing_leg
                self._same_swing_count = 0
            elif current_swing_leg != self._confirmed_swing_leg:
                self._confirmed_swing_leg = current_swing_leg
                self._same_swing_count = 0
            else:
                self._same_swing_count += 1
            self._last_swing_leg = current_swing_leg
        else:
            self._last_swing_leg = None

        count = self._same_swing_count
        if count < _MARCH_RAMP_UP:
            alt_factor = float(count / _MARCH_RAMP_UP)
        elif count < _MARCH_DECAY_START:
            alt_factor = 1.0
        elif count < _MARCH_DECAY_END:
            alt_factor = float(max(0.0, (_MARCH_DECAY_END - count) /
                                   (_MARCH_DECAY_END - _MARCH_DECAY_START)))
        else:
            alt_factor = 0.0

        march_reward = float(np.clip(march_lift * single_support * alt_factor, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 6. Stance [0, 1] — COM on heel-toe line + forward step
        # ------------------------------------------------------------------
        if swing == 0:
            # Right in air → left is stance
            stance_heel = physics.data.site_xpos[physics._sid_foot[2], :2]
            stance_toe = physics.data.site_xpos[physics._sid_foot[3], :2]
            swing_xy = physics.data.xpos[physics._bid_right_foot, :2]
        elif swing == 1:
            stance_heel = physics.data.site_xpos[physics._sid_foot[0], :2]
            stance_toe = physics.data.site_xpos[physics._sid_foot[1], :2]
            swing_xy = physics.data.xpos[physics._bid_left_foot, :2]
        else:
            stance_heel = None

        stance_reward = 0.0
        if stance_heel is not None:
            ht_vec = stance_toe - stance_heel
            ht_len = float(np.linalg.norm(ht_vec))
            if ht_len > 0.02:
                ht_dir = ht_vec / ht_len
                ht_perp = np.array([-ht_dir[1], ht_dir[0]])
                com_xy = physics.com_ground_projection()
                com_to_heel = com_xy - stance_heel

                # Lateral: COM near heel-toe line
                lateral_reward = float(
                    np.clip(1.0 - abs(np.dot(com_to_heel, ht_perp)) / 0.15, 0.0, 1.0))

                # Longitudinal: COM between heel and toe
                long_reward = float(
                    rewards.tolerance(np.dot(com_to_heel, ht_dir),
                                      bounds=(0.0, 0.20), margin=0.15,
                                      value_at_margin=0.2, sigmoid="linear"))

                # Step forward: swing foot ahead of stance
                step_reward = float(
                    rewards.tolerance(np.dot(swing_xy - stance_heel, ht_dir),
                                      bounds=(0.05, 0.30), margin=0.15,
                                      value_at_margin=0.1, sigmoid="linear"))

                stance_reward = float(
                    np.clip(lateral_reward * 0.4 + long_reward * 0.3 +
                            step_reward * 0.3, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 7. Approach [0, 1] — walk toward ball / approach point
        # ------------------------------------------------------------------
        torso_xy = physics.torso_xy()
        ball_xy = physics.ball_xy()
        target_xy = physics.target_xy()
        ball_to_target = target_xy - ball_xy
        b2t_norm = np.linalg.norm(ball_to_target)
        dir_b2t = ball_to_target / b2t_norm if b2t_norm > 1e-6 else np.array([1.0, 0.0])

        # Adaptive approach: aim behind ball when far, at ball when close
        ideal_approach = ball_xy - dir_b2t * _APPROACH_OFFSET
        if np.linalg.norm(ideal_approach - torso_xy) > 0.8:
            approach_point = ideal_approach
        else:
            approach_point = ball_xy

        approach_dist = float(
            rewards.tolerance(np.linalg.norm(approach_point - torso_xy),
                              bounds=(0, _BALL_RADIUS), margin=3.0,
                              value_at_margin=0.1, sigmoid="linear"))

        # Velocity direction toward approach point
        torso_vel = physics.velocity()[:2]
        approach_dir = approach_point - torso_xy
        ad_norm = np.linalg.norm(approach_dir)
        if ad_norm > 1e-6:
            dir_approach = approach_dir / ad_norm
        else:
            dir_approach = np.zeros(2)

        tv_norm = np.linalg.norm(torso_vel)
        if tv_norm > 1e-6 and ad_norm > 1e-6:
            vel_toward = float(max(0.0, np.dot(torso_vel / tv_norm, dir_approach)))
        elif ad_norm <= 1e-6:
            vel_toward = 1.0
        else:
            vel_toward = 0.0

        # Facing direction (torso x-axis toward approach point)
        torso_fwd = physics.data.xmat[physics._bid_torso, :2]
        tf_norm = np.linalg.norm(torso_fwd)
        if tf_norm > 1e-6 and ad_norm > 1e-6:
            facing_toward = float(max(0.0, np.dot(torso_fwd / tf_norm, dir_approach)))
        else:
            facing_toward = 0.0

        approach_reward = float(np.clip(
            0.30 * approach_dist + 0.35 * vel_toward + 0.35 * facing_toward, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 8. Kick [0, 1] — ball velocity toward target
        # ------------------------------------------------------------------
        ball_vel_xy = physics.ball_linear_velocity_xy()
        ball_speed = float(np.linalg.norm(ball_vel_xy))
        if ball_speed > 1e-6:
            cos_angle = float(np.dot(ball_vel_xy / ball_speed, dir_b2t))
            speed_factor = float(np.clip(ball_speed / _BALL_SPEED_SATURATE, 0.0, 1.0))
            kick_reward = float(max(0.0, cos_angle) * speed_factor)
        else:
            kick_reward = 0.0

        # Ball already rolling → approach is maxed (agent should set up next kick)
        if ball_speed > 0.2:
            approach_reward = 1.0

        # ------------------------------------------------------------------
        # 9. Target [0, 1] — ball in target zone
        # ------------------------------------------------------------------
        target_size = physics.get_target_size()
        target_hit = (np.linalg.norm(target_xy - ball_xy) < target_size + _BALL_RADIUS)
        target_reward = 1.0 if target_hit else 0.0

        # Respawn when ball reaches target
        if target_hit:
            self._respawn_after_hit(physics)

        # ------------------------------------------------------------------
        # Combine
        # ------------------------------------------------------------------
        reward = (
            _W_FLAT_FOOT * flat_foot_reward
            + _W_STAND * stand_reward
            + _W_WEIGHT_SHIFT * weight_shift_reward
            + _W_MARCH * march_reward
            + _W_STANCE * stance_reward
            + _W_APPROACH * approach_reward
            + _W_KICK * kick_reward
            + _W_TARGET * target_reward
            # Penalties
            + _W_FEET_UNDER * (feet_under - 1.0)
            + _W_HIP_ALIGN * hip_align
            + _W_LEG_SPREAD * leg_spread
            + _W_SELF_COLLISION * self_collision
            + _W_TORSO_VEL_ALIGN * (torso_vel_align - 1.0)
            + _W_FEET_TORSO_ALIGN * (feet_torso_align - 1.0)
        )

        if target_hit:
            reward += _TARGET_HIT_BONUS

        self._reward_components = {
            "flat_foot": _W_FLAT_FOOT * flat_foot_reward,
            "stand": _W_STAND * stand_reward,
            "weight_shift": _W_WEIGHT_SHIFT * weight_shift_reward,
            "march": _W_MARCH * march_reward,
            "stance": _W_STANCE * stance_reward,
            "approach": _W_APPROACH * approach_reward,
            "kick": _W_KICK * kick_reward,
            "target": _W_TARGET * target_reward,
            "feet_under": _W_FEET_UNDER * (feet_under - 1.0),
            "hip_align": _W_HIP_ALIGN * hip_align,
            "leg_spread": _W_LEG_SPREAD * leg_spread,
            "self_collision": _W_SELF_COLLISION * self_collision,
            "torso_vel_align": _W_TORSO_VEL_ALIGN * (torso_vel_align - 1.0),
            "feet_torso_align": _W_FEET_TORSO_ALIGN * (feet_torso_align - 1.0),
            "target_hit_bonus": _TARGET_HIT_BONUS if target_hit else 0.0,
        }

        return float(reward)
