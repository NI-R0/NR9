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

"""Humanoid_Ball domain: free-floating humanoid with ball-kick task.

Adapted from walker_3D_ball for the full humanoid model. The agent must stand,
walk, and kick the ball into the target zone using only its feet.

Reward design follows the same single-phase approach as walker_3D_ball:
all reward components are directly active from the first step.

Foot design:
  Each foot has two touch sensors (right_right_foot, left_right_foot for the
  right foot; right_left_foot, left_left_foot for the left foot). These serve
  as the foot contact sensors analogous to heel/toe in the walker domain.
  The flat_foot reward uses the foot body orientation (xmat zz) to measure
  sole flatness when in contact with the ground.

Early termination:
  The episode terminates immediately when the agent falls (torso too
  low or non-foot body parts touching the ground), after a short grace
  period.

Reward normalisation: all positive rewards in [0, 1], penalties in [-1, 0].
Kick reward is [0, 1] based on the angle between ball velocity and target
direction — 0 when the ball moves away or sideways, 1 when it moves
directly toward the target.

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
_CONTROL_TIMESTEP = 0.025
_STUCK_CHECK_STEPS = 25
_STUCK_EPSILON = 1e-3
_STAND_HEIGHT = 1.5  # humanoid torso standing height
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
_TERMINATE_HEIGHT = 0.8       # Torso height (m) below which episode ends
_TERMINATE_NON_FOOT = 0.5    # Sum of tanh(non-foot touch) that triggers end
_TERMINATE_GRACE_STEPS = 10  # Steps after reset before termination is active
_TERMINATE_KNEE_HEIGHT = 0.3  # Knee z-position (m) below which knee is "on ground"

# ---------------------------------------------------------------------------
# Reward weights — same structure as walker_3D_ball
# ---------------------------------------------------------------------------
_W_FEET = 0.08
_W_FLAT_FOOT = 0.08
_W_STAND = 0.10
_W_WEIGHT_SHIFT = 0.10
_W_MARCH = 0.10
_W_APPROACH = 0.12
_W_GAIT = 0.10
_W_KICK = 0.17
_W_TARGET = 0.15

assert abs(sum([
    _W_FEET, _W_FLAT_FOOT, _W_STAND, _W_WEIGHT_SHIFT,
    _W_MARCH, _W_APPROACH, _W_GAIT, _W_KICK, _W_TARGET,
]) - 1.0) < 1e-9, "Positive reward weights must sum to 1.0"

_W_EFFORT = 0.01
_W_FEET_UNDER = 0.03
_W_HIP_ALIGN = 0.03
_W_LEG_SPREAD = 0.02
_W_SELF_COLLISION = 0.05

# Normalisation constants
_LEG_SPREAD_THRESHOLD = 0.2
_HIP_YAW_MAX = np.radians(45)
_HIP_ROLL_MAX = np.radians(45)
_GAIT_MIN_VELOCITY = 0.3
_SHIN_LENGTH = 0.25
_MARCH_KNEE_TARGET = np.radians(60)
_MARCH_HIP_PITCH_TARGET = np.radians(45)
_FEET_UNDER_MAX_OFFSET = 0.5
_FLAT_FOOT_TOUCH_THRESHOLD = 0.3
_ANKLE_ROLL_MAX = np.radians(30)

# March alternation
_MARCH_RAMP_UP = 10
_MARCH_DECAY_START = 40
_MARCH_DECAY_END = 60

# Touch sensor names
_NON_FOOT_TOUCHES = (
    "torso_touch",
    "head_touch",
    "lower_waist_touch",
    "butt_touch",
    "right_thigh_touch",
    "right_shin_touch",
    "left_thigh_touch",
    "left_shin_touch",
    "right_upper_arm_touch",
    "right_lower_arm_touch",
    "right_hand_touch",
    "left_upper_arm_touch",
    "left_lower_arm_touch",
    "left_hand_touch",
)
# Foot touch sensors (4 total: 2 per foot)
_FOOT_TOUCHES = (
    "right_right_foot_touch",
    "left_right_foot_touch",
    "right_left_foot_touch",
    "left_left_foot_touch",
)
_ALL_TOUCHES = _NON_FOOT_TOUCHES + _FOOT_TOUCHES

SUITE = containers.TaggedTasks()
FILE = "humanoid_ball.xml"


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Numerically stable logistic sigmoid, output ∈ (0, 1)."""
    z = k * x
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    xml_path = os.path.join(os.path.dirname(__file__), FILE)
    with open(xml_path, "r") as f:
        xml_string = f.read()
    assets = {f"./common/{k}": v for k, v in common.ASSETS.items()}
    return xml_string, assets


def _make_task(time_limit, random, environment_kwargs):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = HumanoidBall(random=random)
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
    """Returns the Kick task: stand, walk to the ball, kick it into the target."""
    return _make_task(time_limit, random, environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Humanoid_Ball domain."""

    def _ensure_indices(self):
        """Precompute MuJoCo name→id indices for fast array access."""
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
        self._bid_right_thigh = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "right_thigh")
        self._bid_left_thigh = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "left_thigh")
        self._bid_right_shin = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "right_shin")
        self._bid_left_shin = mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "left_shin")

        # Adjacent body pairs (exempt from self-collision penalty)
        self._adjacent_pairs = frozenset([
            frozenset((self._bid_torso, self._bid_right_thigh)),
            frozenset((self._bid_torso, self._bid_left_thigh)),
            frozenset((self._bid_right_thigh, self._bid_right_shin)),
            frozenset((self._bid_left_thigh, self._bid_left_shin)),
            frozenset((self._bid_right_shin, self._bid_right_foot)),
            frozenset((self._bid_left_shin, self._bid_left_foot)),
        ])

        # Joint qpos addresses — humanoid has more joints than walker
        joint_names = [
            # Leg joints (12)
            "right_hip_x", "right_hip_z", "right_hip_y",
            "right_knee", "right_ankle_y", "right_ankle_x",
            "left_hip_x", "left_hip_z", "left_hip_y",
            "left_knee", "left_ankle_y", "left_ankle_x",
            # Trunk joints (3)
            "abdomen_x", "abdomen_y", "abdomen_z",
            # Arm joints (10)
            "right_shoulder1", "right_shoulder2", "right_elbow",
            "left_shoulder1", "left_shoulder2", "left_elbow",
            # Note: humanoid has 20 DoF after freejoint, so 22 total
            # We include all actuated joints for observation
        ]
        self._qpos_joints = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, n)]
            for n in joint_names
        ], dtype=np.int64)
        self._qpos_root = 0
        self._qpos_ball = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "ball_joint")
        ]
        self._qvel_ball = model.jnt_dofadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "ball_joint")
        ]

        # Specific joint addresses for reward computation
        self._qpos_r_hip_yaw = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_hip_z")]
        self._qpos_l_hip_yaw = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_hip_z")]
        self._qpos_r_hip_roll = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_hip_x")]
        self._qpos_l_hip_roll = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_hip_x")]
        self._qpos_r_knee = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_knee")]
        self._qpos_l_knee = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_knee")]
        self._qpos_r_hip_pitch = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_hip_y")]
        self._qpos_l_hip_pitch = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_hip_y")]
        self._qpos_r_ankle_roll = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "right_ankle_x")]
        self._qpos_l_ankle_roll = model.jnt_qposadr[
            mujoco.mj_name2id(model, mjt.mjOBJ_JOINT, "left_ankle_x")]

        # Sensor addresses
        self._sensor_linvel = model.sensor_adr[
            mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, "torso_subtreelinvel")]
        self._sensor_foot = np.array([
            model.sensor_adr[
                mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, name)]
            for name in _FOOT_TOUCHES
        ], dtype=np.int64)
        self._si_r_heel = 0  # right_right_foot
        self._si_r_toe = 1   # left_right_foot
        self._si_l_heel = 2  # right_left_foot
        self._si_l_toe = 3   # left_left_foot
        self._sensor_non_foot = np.array([
            model.sensor_adr[
                mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, name)]
            for name in _NON_FOOT_TOUCHES
        ], dtype=np.int64)
        self._sensor_all_touch = np.array([
            model.sensor_adr[
                mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, name)]
            for name in _ALL_TOUCHES
        ], dtype=np.int64)

        # Site IDs for foot touch sensors
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

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        self._ensure_indices()
        return self.data.xmat[self._bid_torso, 8]

    def torso_height(self):
        """Returns the world-z height of the torso."""
        self._ensure_indices()
        return self.data.xpos[self._bid_torso, 2]

    def torso_xy(self):
        """Returns the [x, y] position of the torso."""
        self._ensure_indices()
        return self.data.xpos[self._bid_torso, :2].copy()

    def horizontal_velocity(self):
        """Returns the horizontal speed of the center-of-mass."""
        self._ensure_indices()
        linvel = self.data.sensordata[self._sensor_linvel:self._sensor_linvel + 3]
        return float(np.linalg.norm(linvel[:2]))

    def orientations(self):
        """Returns planar orientations of all non-root bodies."""
        self._ensure_indices()
        return self.data.xmat[1:, [0, 2]].ravel()

    def touch_forces(self):
        """Returns touch forces of all body parts as a 1-D array."""
        self._ensure_indices()
        return np.tanh(self.data.sensordata[self._sensor_all_touch])

    def feet_touch(self):
        """Returns summed tanh-saturated touch force for feet only."""
        self._ensure_indices()
        return float(np.sum(np.tanh(self.data.sensordata[self._sensor_foot])))

    @staticmethod
    def _flatness_curve(x: float) -> float:
        """Linear flatness curve: 0 at vertical, 0.5 at horizontal."""
        x = max(0.0, min(1.0, x))
        return float(0.5 * x)

    def flat_foot_contact(self):
        """Returns the flatness [0, 1] of the best foot touching the ground."""
        self._ensure_indices()
        touches = np.tanh(self.data.sensordata[self._sensor_foot])
        # Sensor order: right_right_foot, left_right_foot, right_left_foot, left_left_foot
        # Right foot sensors: index 0, 1
        # Left foot sensors: index 2, 3
        if ((touches[0] > _FLAT_FOOT_TOUCH_THRESHOLD and touches[1] > _FLAT_FOOT_TOUCH_THRESHOLD) or
            (touches[2] > _FLAT_FOOT_TOUCH_THRESHOLD and touches[3] > _FLAT_FOOT_TOUCH_THRESHOLD)):
            return 1.0

        r_touching = touches[0] > _FLAT_FOOT_TOUCH_THRESHOLD or touches[1] > _FLAT_FOOT_TOUCH_THRESHOLD
        l_touching = touches[2] > _FLAT_FOOT_TOUCH_THRESHOLD or touches[3] > _FLAT_FOOT_TOUCH_THRESHOLD

        # Sole tilt flatness via xmat zz-component
        r_tilt = float(max(0.0, self.data.xmat[self._bid_right_foot, 8])) if r_touching else 0.0
        l_tilt = float(max(0.0, self.data.xmat[self._bid_left_foot, 8])) if l_touching else 0.0

        # Ankle-roll flatness
        r_roll = 0.0
        l_roll = 0.0
        if r_touching:
            r_roll_angle = abs(float(self.data.qpos[self._qpos_r_ankle_roll]))
            r_roll = 1.0 - min(1.0, r_roll_angle / _ANKLE_ROLL_MAX)
        if l_touching:
            l_roll_angle = abs(float(self.data.qpos[self._qpos_l_ankle_roll]))
            l_roll = 1.0 - min(1.0, l_roll_angle / _ANKLE_ROLL_MAX)

        r_flat = self._flatness_curve(r_tilt) * self._flatness_curve(r_roll) if r_touching else 0.0
        l_flat = self._flatness_curve(l_tilt) * self._flatness_curve(l_roll) if l_touching else 0.0
        return max(r_flat, l_flat)

    def foot_contact_points(self):
        """Returns the xy positions of all foot contact points on the ground."""
        self._ensure_indices()
        touches = np.tanh(self.data.sensordata[self._sensor_foot])
        in_contact = touches > _FLAT_FOOT_TOUCH_THRESHOLD
        if not np.any(in_contact):
            return np.empty((0, 2))
        return self.data.site_xpos[self._sid_foot[in_contact], :2].copy()

    def com_ground_projection(self):
        """Returns the [x, y] ground projection of the centre of mass."""
        self._ensure_indices()
        return self.data.subtree_com[self._bid_torso, :2].copy()

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
        """Returns the convex hull of a set of 2D points."""
        pts = np.asarray(points)
        n = len(pts)
        if n <= 2:
            return pts.copy()

        start = int(np.argmin(pts[:, 0]))
        hull_indices = []
        p = start
        while True:
            hull_indices.append(p)
            q = (p + 1) % n
            for i in range(n):
                if i == p:
                    continue
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
        """Returns distance from COM ground projection to support polygon."""
        contact_pts = self.foot_contact_points()
        if len(contact_pts) < 1:
            return 1e6
        if len(contact_pts) == 1:
            return float(np.linalg.norm(self.com_ground_projection() - contact_pts[0]))

        com_xy = self.com_ground_projection()

        if len(contact_pts) == 2:
            p1, p2 = contact_pts
            seg = p2 - p1
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-12:
                return float(np.linalg.norm(com_xy - p1))
            t = np.clip(np.dot(com_xy - p1, seg) / seg_len_sq, 0.0, 1.0)
            proj = p1 + t * seg
            return float(np.linalg.norm(com_xy - proj))

        hull_pts = self._convex_hull_2d(contact_pts)
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
        """Moves the mocap target body to (x, y, 0.1)."""
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
        """Returns [right_hip_yaw, left_hip_yaw] joint angles."""
        self._ensure_indices()
        return np.array([
            self.data.qpos[self._qpos_r_hip_yaw],
            self.data.qpos[self._qpos_l_hip_yaw],
        ]).ravel()

    def hip_roll_angles(self):
        """Returns [right_hip_roll, left_hip_roll] joint angles."""
        self._ensure_indices()
        return np.array([
            self.data.qpos[self._qpos_r_hip_roll],
            self.data.qpos[self._qpos_l_hip_roll],
        ]).ravel()

    def foot_heights(self):
        """Returns [right_foot_z, left_foot_z] world-z heights."""
        self._ensure_indices()
        return np.array([
            self.data.xpos[self._bid_right_foot, 2],
            self.data.xpos[self._bid_left_foot, 2],
        ])

    def knee_heights(self):
        """Returns [right_knee_z, left_knee_z] world-z heights."""
        self._ensure_indices()
        return np.array([
            self.data.xpos[self._bid_right_shin, 2],
            self.data.xpos[self._bid_left_shin, 2],
        ])

    def joint_positions(self):
        """Returns all non-root joint angles as a 1-D array."""
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
        """Returns [right_hip_pitch, left_hip_pitch] joint angles."""
        self._ensure_indices()
        return np.array([
            self.data.qpos[self._qpos_r_hip_pitch],
            self.data.qpos[self._qpos_l_hip_pitch],
        ]).ravel()

    def com_lateral_offset(self):
        """Returns the lateral (y) distance from COM to feet midpoint."""
        self._ensure_indices()
        com_y = self.data.subtree_com[self._bid_torso, 1]
        right_foot_y = self.data.xpos[self._bid_right_foot, 1]
        left_foot_y = self.data.xpos[self._bid_left_foot, 1]
        feet_mid_y = (right_foot_y + left_foot_y) / 2.0
        return float(com_y - feet_mid_y)

    def com_lateral_to_foot(self):
        """Returns [com_to_right_foot_y, com_to_left_foot_y] distances."""
        self._ensure_indices()
        com_y = self.data.subtree_com[self._bid_torso, 1]
        right_foot_y = self.data.xpos[self._bid_right_foot, 1]
        left_foot_y = self.data.xpos[self._bid_left_foot, 1]
        return np.array([com_y - right_foot_y, com_y - left_foot_y])

    def feet_xy_offset(self):
        """Returns xy-distance from feet midpoint to COM."""
        self._ensure_indices()
        com_xy = self.data.subtree_com[self._bid_torso, :2]
        right_foot_xy = self.data.xpos[self._bid_right_foot, :2]
        left_foot_xy = self.data.xpos[self._bid_left_foot, :2]
        right_foot_com = float(np.linalg.norm(com_xy - right_foot_xy))
        left_foot_com = float(np.linalg.norm(com_xy - left_foot_xy))
        return float((right_foot_com + left_foot_com) / 2)

    def self_collision_penalty(self):
        """Returns the total self-collision penalty [-1, 0]."""
        self._ensure_indices()
        data = self.data
        model = self.model.ptr
        depth_scale = 0.05

        total_depth = 0.0
        for i in range(int(data.ncon)):
            c = data.contact[i]
            geom1_body = model.geom_bodyid[int(c.geom1)]
            geom2_body = model.geom_bodyid[int(c.geom2)]

            if geom1_body == geom2_body:
                continue

            pair = frozenset((geom1_body, geom2_body))
            if pair in self._adjacent_pairs:
                continue

            if geom1_body == self._bid_ball or geom2_body == self._bid_ball:
                continue
            if geom1_body == self._bid_target or geom2_body == self._bid_target:
                continue
            if geom1_body == 0 or geom2_body == 0:
                continue

            if c.dist < 0:
                total_depth += abs(c.dist)

        return float(-min(1.0, total_depth / depth_scale))


class HumanoidBall(base.Task):
    """Humanoid with ball-kick task (feet only).

    All reward components are directly active from the first step.
    The agent learns standing, walking, and kicking in parallel.
    """

    def __init__(self, random=None):
        self._target_size = _TARGET_SIZE_MAX
        self._consecutive_successes = 0
        self._target_pos = None
        self._reward_components: dict[str, float] = {}
        self._last_swing_leg: str | None = None
        self._same_swing_count: int = 0
        self._step_count = 0
        super().__init__(random=random)

    def register_success(self):
        """Call when the agent successfully hits the target during evaluation."""
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
        """Sets the state of the environment at the start of each episode."""
        self._setup_episode(physics)
        super().initialize_episode(physics)

    def _setup_episode(self, physics):
        """Reset humanoid pose, ball, target, and per-episode bookkeeping."""
        spawn_x = self.random.uniform(-1.5, 1.5)
        spawn_y = self.random.uniform(-1.5, 1.5)
        physics.named.data.qpos["root"] = [spawn_x, spawn_y, 1.8, 1.0, 0.0, 0.0, 0.0]
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
        self._prev_action = None

        self._step_count = 0
        self._last_swing_leg = None
        self._same_swing_count = 0

    def _place_target(self, physics):
        """Randomly places the target at a random angle and distance."""
        angle = self.random.uniform(0, 2 * np.pi)
        dist = self.random.uniform(_TARGET_MIN_DIST, _TARGET_MAX_DIST)
        self._target_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        physics.set_target_position(self._target_pos)

    def _respawn_after_hit(self, physics):
        """Full mid-episode respawn: walker, ball, and target are reset."""
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
        """Returns True when the agent has fallen."""
        self._step_count += 1
        if self._step_count <= _TERMINATE_GRACE_STEPS:
            return False
        if physics.torso_height() < _TERMINATE_HEIGHT:
            return True
        if physics.non_foot_touch() > _TERMINATE_NON_FOOT:
            return True
        knee_z = physics.knee_heights()
        if knee_z[0] < _TERMINATE_KNEE_HEIGHT or knee_z[1] < _TERMINATE_KNEE_HEIGHT:
            return True
        if getattr(self, '_stuck_count', 0) >= _STUCK_CHECK_STEPS:
            logger.warning("Episode terminated: agent stuck")
            return True
        return False

    def get_reward(self, physics):
        """Direct reward: all components active, no gates."""
        ctrl = physics.control()

        # ======================================================================
        # Shared quantities
        # ======================================================================
        feet_touch_raw = physics.feet_touch()
        non_foot_touch_raw = physics.non_foot_touch()

        feet_contact = _sigmoid(feet_touch_raw - 0.5, k=4.0)
        non_foot_contact = _sigmoid(non_foot_touch_raw - 0.5, k=4.0)
        feet_only = 1.0 - non_foot_contact

        flat_foot_reward = physics.flat_foot_contact()
        feet_reward = feet_contact * feet_only

        effort_penalty = -float(np.mean(ctrl**2))
        self_collision = physics.self_collision_penalty()

        standing = float(np.clip(physics.torso_height() / _STAND_HEIGHT, 0.0, 1.0))
        upright = float((1 + physics.torso_upright()) / 2)
        stand_reward = standing * upright

        hip_yaw = physics.hip_yaw_angles()
        hip_roll = physics.hip_roll_angles()
        hip_align_penalty = -float(
            np.clip(
                (
                    np.mean(np.abs(hip_yaw) / _HIP_YAW_MAX)
                    + np.mean(np.abs(hip_roll) / _HIP_ROLL_MAX)
                )
                / 2.0,
                0.0,
                1.0,
            )
        )

        feet_dist = physics.feet_horizontal_distance()
        leg_spread = -float(
            np.clip(
                (feet_dist - _LEG_SPREAD_THRESHOLD) / _LEG_SPREAD_THRESHOLD, 0.0, 1.0
            )
        )

        feet_offset = physics.feet_xy_offset()
        feet_under = 1.0 - float(
            np.clip(feet_offset / _FEET_UNDER_MAX_OFFSET, 0.0, 1.0)
        )

        # ======================================================================
        # Weight shift reward
        # ======================================================================
        com_to_feet = physics.com_lateral_to_foot()
        half_foot_dist = max(physics.feet_lateral_distance() / 2.0, 1e-6)
        shift_right = 1.0 - float(
            np.clip(np.abs(com_to_feet[0]) / half_foot_dist, 0.0, 1.0)
        )
        shift_left = 1.0 - float(
            np.clip(np.abs(com_to_feet[1]) / half_foot_dist, 0.0, 1.0)
        )
        weight_shift_reward = float(np.clip(max(shift_right, shift_left), 0.0, 1.0))
        stillness = 1.0 - float(
            np.clip(physics.horizontal_velocity() / _GAIT_MIN_VELOCITY, 0.0, 1.0)
        )

        # ======================================================================
        # March reward
        # ======================================================================
        knee = physics.knee_angles()
        hip_pitch = physics.hip_pitch_angles()
        foot_z = physics.foot_heights()

        right_knee_lift = float(np.clip(-knee[0] / _MARCH_KNEE_TARGET, 0.0, 1.0))
        left_knee_lift = float(np.clip(-knee[1] / _MARCH_KNEE_TARGET, 0.0, 1.0))
        right_hip_lift = float(
            np.clip(hip_pitch[0] / _MARCH_HIP_PITCH_TARGET, 0.0, 1.0)
        )
        left_hip_lift = float(
            np.clip(hip_pitch[1] / _MARCH_HIP_PITCH_TARGET, 0.0, 1.0)
        )

        right_lift = (right_knee_lift + right_hip_lift) / 2.0
        left_lift = (left_knee_lift + left_hip_lift) / 2.0

        touch_r = float(np.tanh(physics.data.sensordata[
            physics._sensor_foot[physics._si_r_heel]]))
        touch_l = float(np.tanh(physics.data.sensordata[
            physics._sensor_foot[physics._si_l_heel]]))
        touch_sum = touch_r + touch_l
        single_support = float(
            rewards.tolerance(
                touch_sum,
                bounds=(0.7, 1.3),
                margin=1.0,
                value_at_margin=0.1,
                sigmoid="linear",
            )
        )

        right_as_swing = right_lift * (1.0 - touch_r) * touch_l
        left_as_swing = left_lift * (1.0 - touch_l) * touch_r
        march_lift = float(max(right_as_swing, left_as_swing))

        # March alternation
        if right_as_swing > left_as_swing:
            current_swing_leg = "right"
        else:
            current_swing_leg = "left"

        if self._last_swing_leg is None or current_swing_leg != self._last_swing_leg:
            self._same_swing_count = 0
        else:
            self._same_swing_count += 1
        self._last_swing_leg = current_swing_leg

        count = self._same_swing_count
        if count < _MARCH_RAMP_UP:
            alternation_factor = float(count / _MARCH_RAMP_UP)
        elif count < _MARCH_DECAY_START:
            alternation_factor = 1.0
        elif count < _MARCH_DECAY_END:
            alternation_factor = float(
                max(0.0, (_MARCH_DECAY_END - count) / (_MARCH_DECAY_END - _MARCH_DECAY_START))
            )
        else:
            alternation_factor = 0.0

        march_reward = float(np.clip(march_lift * single_support * alternation_factor, 0.0, 1.0))

        # ======================================================================
        # Approach + gait reward
        # ======================================================================
        torso_xy = physics.torso_xy()
        ball_xy = physics.ball_xy()
        target_xy = physics.target_xy()
        ball_to_target = target_xy - ball_xy
        ball_to_target_norm = np.linalg.norm(ball_to_target)
        if ball_to_target_norm > 1e-6:
            dir_ball_to_target = ball_to_target / ball_to_target_norm
        else:
            dir_ball_to_target = np.array([1.0, 0.0])

        approach_point = ball_xy - dir_ball_to_target * _APPROACH_OFFSET
        dist_to_approach = np.linalg.norm(approach_point - torso_xy)
        approach_reward = float(
            rewards.tolerance(
                dist_to_approach,
                bounds=(0, _BALL_RADIUS),
                margin=3.0,
                value_at_margin=0.1,
                sigmoid="linear",
            )
        )

        h_vel = physics.horizontal_velocity()
        gait_gate = float(np.clip(h_vel / _GAIT_MIN_VELOCITY, 0.0, 1.0))
        support_foot_z = float(np.min(foot_z))
        swing_foot_lift = float(np.max(foot_z)) - support_foot_z
        foot_clearance = float(
            rewards.tolerance(
                swing_foot_lift,
                bounds=(_SHIN_LENGTH, float("inf")),
                margin=_SHIN_LENGTH,
                value_at_margin=0.1,
                sigmoid="linear",
            )
        )
        gait_reward = gait_gate * (foot_clearance + single_support) / 2.0

        # ======================================================================
        # Kick + target reward
        # ======================================================================
        ball_vel_xy = physics.ball_linear_velocity_xy()
        ball_speed = float(np.linalg.norm(ball_vel_xy))
        if ball_speed > 1e-6:
            ball_vel_dir = ball_vel_xy / ball_speed
            cos_angle = float(np.dot(ball_vel_dir, dir_ball_to_target))
            kick_reward = float(max(0.0, cos_angle))
        else:
            kick_reward = 0.0

        target_size = physics.get_target_size()
        dist_ball_to_target = np.linalg.norm(target_xy - ball_xy)
        target_hit = dist_ball_to_target < target_size + _BALL_RADIUS
        target_reward = 1.0 if target_hit else 0.0

        if target_hit:
            self._respawn_after_hit(physics)

        # ======================================================================
        # Final reward
        # ======================================================================
        reward = (
            _W_FEET * feet_reward
            + _W_FLAT_FOOT * flat_foot_reward
            + _W_STAND * stand_reward
            + _W_WEIGHT_SHIFT * weight_shift_reward * stillness
            + _W_MARCH * march_reward
            + _W_APPROACH * approach_reward
            + _W_GAIT * gait_reward
            + _W_KICK * kick_reward
            + _W_TARGET * target_reward
            + _W_EFFORT * effort_penalty
            + _W_SELF_COLLISION * self_collision
            + _W_FEET_UNDER * (feet_under - 1.0)
            + _W_HIP_ALIGN * hip_align_penalty
            + _W_LEG_SPREAD * leg_spread
        )

        if target_hit:
            reward += _TARGET_HIT_BONUS

        self._reward_components = {
            "feet": _W_FEET * feet_reward,
            "flat_foot": _W_FLAT_FOOT * flat_foot_reward,
            "stand": _W_STAND * stand_reward,
            "weight_shift": _W_WEIGHT_SHIFT * weight_shift_reward * stillness,
            "march": _W_MARCH * march_reward,
            "approach": _W_APPROACH * approach_reward,
            "gait": _W_GAIT * gait_reward,
            "kick": _W_KICK * kick_reward,
            "target": _W_TARGET * target_reward,
            "effort": _W_EFFORT * effort_penalty,
            "self_collision": _W_SELF_COLLISION * self_collision,
            "feet_under": _W_FEET_UNDER * (feet_under - 1.0),
            "hip_align": _W_HIP_ALIGN * hip_align_penalty,
            "leg_spread": _W_LEG_SPREAD * leg_spread,
            "target_hit_bonus": _TARGET_HIT_BONUS if target_hit else 0.0,
        }

        # --- Stuck detection ---
        current_obs = self.get_observation(physics)
        flat_obs = np.concatenate([v.ravel() for v in current_obs.values()])
        current_act = ctrl.copy()

        last_obs = getattr(self, '_last_obs', None)
        last_act = getattr(self, '_last_action', None)
        stuck_count = getattr(self, '_stuck_count', 0)

        if last_obs is not None:
            if (np.max(np.abs(flat_obs - last_obs)) < _STUCK_EPSILON and
                np.max(np.abs(current_act - last_act)) < _STUCK_EPSILON):
                stuck_count += 1
            else:
                stuck_count = 0
        else:
            stuck_count = 0

        self._stuck_count = stuck_count
        self._last_obs = flat_obs
        self._last_action = current_act

        self._prev_action = ctrl.copy()
        return float(reward)