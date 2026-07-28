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

Single-phase reward with cascading smoothed gates:
  All reward components are always active, but each is gated by a
  smoothed rolling mean of the *previous* component's gated value.
  This lets the agent naturally progress from standing to walking to
  kicking without hard phase switches, and prevents it from skipping
  or forgetting earlier skills.

  Gate cascade (each = clamp(mean(prev_gated, last 10 steps), 0, 1)):
    gate_stand    ← feet_reward
    gate_ws       ← stand_reward * gate_stand
    gate_march    ← weight_shift_reward * gate_ws
    gate_approach ← march_reward * gate_march
    gate_full     ← approach_reward * gate_approach

  Stand-specific penalties (hip_align, leg_spread, feet_under,
  symmetry) fade out via ``(1 - gate_march)`` so they don't block
  walking or kicking.

Most reward components are normalised to [0, 1] (rewards) or [-1, 0]
(penalties).  The exception is the *kick* reward, which is in [-1, 1]
so that a ball rolling *away* from the target actively lowers the
reward.  Positive weights sum to 1.0 so a perfect step (with ball
moving toward the target) yields reward = 1.0.  Penalty weights are on
top of the 1.0 budget, so the realistic optimum (task fulfilled with
some unavoidable control cost) lands slightly below 1.0.

Logged reward components are **raw (unweighted)** values in [0, 1] or
[-1, 0] (kick in [-1, 1]), making it easy to inspect each sub-reward's
quality directly.

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
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np


_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.025
_STAND_HEIGHT = 1.6  # top of torso capsule; requires fully upright stance
_WALK_SPEED = 1
_RUN_SPEED = 8
_BALL_RADIUS = 0.2
_BALL_START_POS = np.array([1.5, 0.0, 0.15])
_TARGET_MIN_DIST = 2.0
_TARGET_MAX_DIST = 5.0
_TARGET_SIZE_MAX = 1.0
_TARGET_SIZE_MIN = 0.2
_TARGET_SHRINK = 0.1
_SUCCESS_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Reward design – cascading gates, single set of weights
# ---------------------------------------------------------------------------
# Every component is normalised to [0, 1] (rewards) or [-1, 0] (penalties),
# **except** the kick reward which is in [-1, 1]: a ball moving toward the
# target yields positive values, a ball moving away yields negative values.
#
# **Positive weights** sum to exactly 1.0 → a perfect step (all rewards = 1,
# all penalties = 0, ball moving toward target) yields reward = 1.0.
#
# **Negative (penalty) weights** are *on top* of the 1.0 budget.  They
# pull the reward below 1.0 by an amount proportional to the penalty
# magnitude × weight.  This means the realistic optimum (task fulfilled
# with some unavoidable effort/control cost) lands slightly below 1.0,
# while the theoretical maximum is 1.0.
#
# Gate cascade (smoothed over last _GATE_SMOOTHING steps):
#   gate_stand    ← mean(feet_gated_history)       [feet_reward]
#   gate_ws       ← mean(stand_gated_history)       [stand * gate_stand]
#   gate_march    ← mean(ws_gated_history)          [ws * gate_ws]
#   gate_approach ← mean(march_gated_history)       [march * gate_march]
#   gate_full     ← mean(approach_gated_history)    [approach * gate_approach]
#
# Stand-specific penalties fade via (1 - gate_march) so they don't
# interfere with walking or kicking.
# ---------------------------------------------------------------------------

# Positive weights (sum = 1.0)
_W_FEET = 0.15
_W_STAND = 0.20
_W_SYMMETRY = 0.10
_W_WEIGHT_SHIFT = 0.15
_W_MARCH = 0.10
_W_APPROACH = 0.10
_W_GAIT = 0.10
_W_KICK = 0.05
_W_TARGET = 0.05

# Penalty weights (on top of the 1.0 budget, kept small: 0.01–0.05)
_W_EFFORT = 0.03
_W_FEET_UNDER = 0.03      # fades via (1 - gate_march)
_W_HIP_ALIGN = 0.03       # fades via (1 - gate_march)
_W_LEG_SPREAD = 0.02      # fades via (1 - gate_march)
_W_ANKLE_FLAT = 0.02      # fades via (1 - gate_march)
_W_SMOOTHNESS = 0.02      # always active

# Gate smoothing window (steps)
_GATE_SMOOTHING = 10

# Normalisation constants
_LEG_SPREAD_THRESHOLD = 0.2
_HIP_YAW_MAX = np.radians(45)  # Max hip-yaw range for normalization
_HIP_ROLL_MAX = np.radians(45)  # Max hip-roll deviation from neutral for normalization
# Symmetry: joint indices whose sign must be flipped when comparing left vs
# right, because the joint axes are NOT mirrored in the XML (e.g. hip_roll
# uses axis="1 0 0" for both legs).  Indices refer to the 5-joint per-leg
# block returned by Physics.joint_positions():
#   0=hip_yaw, 1=hip_roll, 2=hip_pitch, 3=knee, 4=ankle
_SYMMETRY_MIRROR_INDICES = (1,)  # hip_roll must be sign-flipped for comparison
_SYMMETRY_JOINT_MAX = np.radians(45)  # Normalization for symmetry reward
_GAIT_MIN_VELOCITY = 0.3  # Min horizontal velocity (m/s) for gait reward to activate
_SHIN_LENGTH = 0.25  # Shin (leg) capsule half-length in m; used as foot-clearance target
_MARCH_KNEE_TARGET = np.radians(60)  # Target knee lift angle for marching
_MARCH_HIP_PITCH_TARGET = np.radians(45)  # Target hip pitch for knee lift
_FEET_UNDER_MAX_OFFSET = (
    0.5  # Max xy-distance (m) from feet-midpoint to torso for feet_under reward
)
_ANKLE_FLAT_MAX = np.radians(30)  # Ankle angle at which flat-foot bonus reaches 0

# Alternation parameters (march + weight_shift)
_MARCH_SWITCH_BONUS = 0.1  # Bonus for switching swing leg (small, just a nudge)
_MARCH_MIN_SAME = 15  # Min steps on same leg before a switch bonus is given
_MARCH_MAX_SAME = 40  # Steps before same-leg reward starts decaying (~1s)
_WEIGHT_SHIFT_SWITCH_BONUS = 0.1
_WEIGHT_SHIFT_MIN_SAME = 15
_WEIGHT_SHIFT_MAX_SAME = 40

# Touch sensor names for feet reward
_NON_FOOT_TOUCHES = (
    "torso_touch",
    "right_thigh_touch",
    "left_thigh_touch",
    "right_leg_touch",
    "left_leg_touch",
)
_FOOT_TOUCHES = (
    "right_foot_touch",
    "left_foot_touch",
)
_ALL_TOUCHES = _NON_FOOT_TOUCHES + _FOOT_TOUCHES

SUITE = containers.TaggedTasks()
FILE = "walker_3D_ball.xml"


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Numerically stable logistic sigmoid, output ∈ (0, 1).

    ``k`` controls steepness.  Used for smooth [0, 1] reward shaping
    instead of hard binary thresholds.
    """
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


def _make_task(move_speed, time_limit, random, environment_kwargs):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Walker3DBall(move_speed=move_speed, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand+Kick task (move_speed=0 → focus on standing first)."""
    return _make_task(0, time_limit, random, environment_kwargs)


@SUITE.add("benchmarking")
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk+Kick task."""
    return _make_task(_WALK_SPEED, time_limit, random, environment_kwargs)


@SUITE.add("benchmarking")
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run+Kick task."""
    return _make_task(_RUN_SPEED, time_limit, random, environment_kwargs)


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

        # Joint qpos / qvel addresses
        joint_names = [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle",
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle",
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
        self._sensor_r_foot = model.sensor_adr[
            mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, "right_foot_touch")]
        self._sensor_l_foot = model.sensor_adr[
            mujoco.mj_name2id(model, mjt.mjOBJ_SENSOR, "left_foot_touch")]
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
        """Returns planar orientations of all bodies.

        For the 3D walker the full rotation matrix carries more information, but
        to keep the observation dimension manageable we return the same projection
        components (xx, xz) used by the planar walker for every non-torso body.
        """
        self._ensure_indices()
        return self.data.xmat[1:, [0, 2]].ravel()

    def touch_forces(self):
        """Returns touch forces of all body parts (including feet) as a 1-D array.

        Each force is passed through ``tanh(force)`` so that light grazes
        produce ~0 while firm contacts saturate to ~1.  Consistent with
        ``feet_touch`` and ``non_foot_touch`` (no ``-3`` offset).
        """
        self._ensure_indices()
        return np.tanh(self.data.sensordata[self._sensor_all_touch])

    def feet_touch(self):
        """Returns summed tanh-saturated touch force for feet only."""
        self._ensure_indices()
        s = self.data.sensordata
        return float(np.tanh(s[self._sensor_r_foot]) +
                      np.tanh(s[self._sensor_l_foot]))

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

    def joint_positions(self):
        """Returns all non-root joint angles as a 1-D array (10 joints).

        Order: right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee,
        right_ankle, left_hip_yaw, left_hip_roll, left_hip_pitch,
        left_knee, left_ankle.
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

    def com_lateral_offset(self):
        """Returns the lateral (y) distance from COM to the midpoint of the feet.

        Positive = COM shifted toward the left foot, negative = toward the right.
        Used for the weight-shift reward: the agent should shift its COM over
        one foot, then the other.
        """
        self._ensure_indices()
        com_y = self.data.subtree_com[self._bid_torso, 1]
        right_foot_y = self.data.xpos[self._bid_right_foot, 1]
        left_foot_y = self.data.xpos[self._bid_left_foot, 1]
        feet_mid_y = (right_foot_y + left_foot_y) / 2.0
        return float(com_y - feet_mid_y)

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
        """Returns the xy-distance from the feet midpoint to the centre of mass.

        A value near 0 means the feet are directly under the COM (correct
        support posture).  Uses ``subtree_com["torso"]`` (the actual centre
        of mass of the torso subtree) rather than ``xpos["torso"]`` so the
        feet must be under the **centre of mass**, not merely under the
        torso body origin.  Used by the ``feet_under`` reward component.
        """
        self._ensure_indices()
        com_xy = self.data.subtree_com[self._bid_torso, :2]
        right_foot_xy = self.data.xpos[self._bid_right_foot, :2]
        left_foot_xy = self.data.xpos[self._bid_left_foot, :2]
        right_foot_com = float(np.linalg.norm(com_xy - right_foot_xy))
        left_foot_com = float(np.linalg.norm(com_xy - left_foot_xy))
        return float((right_foot_com + left_foot_com) / 2)


class Walker3DBall(base.Task):
    """3D walker with cascading-gate reward and target curriculum.

    All reward components are always active, but each is gated by a
    smoothed rolling mean of the previous component's gated value.  This
    lets the agent naturally progress from standing to walking to kicking
    without hard phase switches, and prevents it from skipping or
    forgetting earlier skills.

    Gate cascade (each = clamp(mean(prev_gated, last 10 steps), 0, 1)):
      gate_stand    ← feet_reward
      gate_ws       ← stand_reward * gate_stand
      gate_march    ← weight_shift_reward * gate_ws
      gate_approach ← march_reward * gate_march
      gate_full     ← approach_reward * gate_approach

    Stand-specific penalties (hip_align, leg_spread, feet_under,
    symmetry) fade out via ``(1 - gate_march)`` so they don't block
    walking or kicking.

    Target curriculum: ``register_success`` increments a counter.  After
    ``_SUCCESS_THRESHOLD`` consecutive successes the target zone shrinks
    by ``_TARGET_SHRINK`` (down to ``_TARGET_SIZE_MIN``).  A failure
    resets the consecutive-success counter.
    """

    def __init__(self, move_speed, random=None):
        """Initializes an instance of `Walker3DBall`.

        Args:
          move_speed: A float. If zero, the stand reward dominates. Otherwise this
            specifies a target horizontal velocity for the approach reward.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a
            seed automatically (default).
        """
        self._move_speed = move_speed
        self._target_size = _TARGET_SIZE_MAX
        self._consecutive_successes = 0
        self._target_pos = None
        self._reward_components: dict[str, float] = {}
        self._prev_action: np.ndarray | None = None  # for action smoothness penalty
        self._last_swing_leg: str | None = (
            None  # 'right' or 'left' (for march alternation)
        )
        self._same_swing_count: int = 0  # consecutive steps with same swing leg
        self._last_shift_side: str | None = (
            None  # 'right' or 'left' (for weight-shift alternation)
        )
        self._same_shift_count: int = 0  # consecutive steps with same shift side
        # Gate history: rolling window of gated values for smoothing.
        # Each entry is the *gated* (i.e. gate × raw) value of the
        # corresponding component at that step.
        self._gate_history: dict[str, collections.deque] = {
            "feet": collections.deque(maxlen=_GATE_SMOOTHING),
            "stand": collections.deque(maxlen=_GATE_SMOOTHING),
            "weight_shift": collections.deque(maxlen=_GATE_SMOOTHING),
            "march": collections.deque(maxlen=_GATE_SMOOTHING),
            "approach": collections.deque(maxlen=_GATE_SMOOTHING),
        }
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
        physics.named.data.qpos["root"] = [0.0, 0.0, 1.3, 1.0, 0.0, 0.0, 0.0]
        physics.named.data.qvel["root"] = 0.0
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        # Restore upright root orientation – randomize_limited_and_rotational_joints
        # also randomizes the free-joint quaternion, overwriting the upright pose.
        root_qpos = physics.named.data.qpos["root"].copy()
        root_qpos[3:] = [1.0, 0.0, 0.0, 0.0]
        physics.named.data.qpos["root"] = root_qpos

        physics.named.data.qpos["ball_joint"] = list(_BALL_START_POS) + [1, 0, 0, 0]
        physics.named.data.qvel["ball_joint"] = 0.0

        physics.set_target_size(self._target_size)
        self._place_target(physics)
        self._prev_action = None  # reset action history
        self._last_swing_leg = None
        self._same_swing_count = 0
        self._last_shift_side = None
        self._same_shift_count = 0
        # Reset gate history at the start of each episode
        for key in self._gate_history:
            self._gate_history[key].clear()

        super().initialize_episode(physics)

    def _place_target(self, physics):
        """Randomly places the target at a random angle and distance."""
        angle = self.random.uniform(0, 2 * np.pi)
        dist = self.random.uniform(_TARGET_MIN_DIST, _TARGET_MAX_DIST)
        self._target_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        physics.set_target_position(self._target_pos)

    def _reset_ball_and_target(self, physics):
        """Re-places ball and target after a successful hit (mid-episode)."""
        physics.named.data.qpos["ball_joint"] = list(_BALL_START_POS) + [1, 0, 0, 0]
        physics.named.data.qvel["ball_joint"] = 0.0
        self._place_target(physics)

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

    def _gate_mean(self, key: str) -> float:
        """Return the smoothed (rolling mean) gated value for a gate key.

        Returns 0.0 if the history is empty (start of episode).
        """
        hist = self._gate_history[key]
        if not hist:
            return 0.0
        return float(sum(hist) / len(hist))

    def get_reward(self, physics):
        """Cascading-gate reward: all components active, each gated by the
        smoothed rolling mean of the previous component's gated value.

        Positive weights sum to 1.0 → perfect step = 1.0.
        Penalty weights are on top → realistic optimum < 1.0.

        ``_reward_components`` stores **raw (unweighted)** values so logged
        components directly show each sub-reward's quality in [0, 1] or
        [-1, 0] (kick in [-1, 1]).
        """
        ctrl = physics.control()

        # ======================================================================
        # Shared quantities (computed once)
        # ======================================================================
        feet_touch_raw = physics.feet_touch()
        non_foot_touch_raw = physics.non_foot_touch()

        # --- Smooth [0, 1] contact signals via sigmoid ---
        feet_contact = _sigmoid(feet_touch_raw - 0.5, k=4.0)
        non_foot_contact = _sigmoid(non_foot_touch_raw - 0.5, k=4.0)

        # feet_only: 1 when no non-foot contact, 0 when any. ∈ [0, 1].
        feet_only = 1.0 - non_foot_contact

        # --- Flat-foot signal [0, 1]: 1 when ankles near neutral (flat feet),
        # 0 when strongly plantarflexed (toe-stand).  Indices 4 and 9 in
        # joint_positions() are right_ankle and left_ankle respectively.
        # Used as a weak penalty (toe-stand → penalty) that fades once marching.
        joints = physics.joint_positions()
        ankle_angles = np.array([joints[4], joints[9]])
        ankle_flat = 1.0 - float(
            np.clip(np.mean(np.abs(ankle_angles)) / _ANKLE_FLAT_MAX, 0.0, 1.0)
        )

        # --- Feet reward [0, 1]: foot contact scaled by absence of non-foot contact ---
        feet_reward = feet_contact * feet_only

        # --- Effort penalty [-1, 0]: mean(ctrl^2) is in [0, 1] (ctrl ∈ [-1,1]) ---
        effort_penalty = -float(np.mean(ctrl**2))

        # --- Standing [0, 1]: height + upright orientation ---
        standing = float(np.clip(physics.torso_height() / _STAND_HEIGHT, 0.0, 1.0))
        upright = float((1 + physics.torso_upright()) / 2)
        stand_reward = (3 * standing + upright) / 4  # in [0, 1]

        # --- Hip alignment penalty [-1, 0] ---
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

        # --- Leg spread penalty [-1, 0] ---
        feet_dist = physics.feet_horizontal_distance()
        leg_spread = -float(
            np.clip(
                (feet_dist - _LEG_SPREAD_THRESHOLD) / _LEG_SPREAD_THRESHOLD, 0.0, 1.0
            )
        )

        # --- Symmetry reward [0, 1] ---
        # hip_roll has the same axis ("1 0 0") for both legs, so a physically
        # symmetric stance means right_hip_roll == -left_hip_roll.  We flip the
        # sign of those joints before comparing so the reward matches the true
        # symmetric pose.
        # joints already computed above for ankle_flat
        right_joints = joints[:5].copy()
        left_joints = joints[5:]
        right_joints[list(_SYMMETRY_MIRROR_INDICES)] *= -1.0
        symmetry_err = float(
            np.mean(np.abs(right_joints - left_joints) / _SYMMETRY_JOINT_MAX)
        )
        symmetry_reward = 1.0 - float(np.clip(symmetry_err, 0.0, 1.0))

        # --- Smoothness penalty [-1, 0] ---
        if self._prev_action is not None:
            action_diff = float(np.mean((ctrl - self._prev_action) ** 2) / 4.0)
        else:
            action_diff = 0.0
        smoothness_penalty = -float(np.clip(action_diff, 0.0, 1.0))

        # --- Feet under torso [0, 1] ---
        feet_offset = physics.feet_xy_offset()
        feet_under = 1.0 - float(
            np.clip(feet_offset / _FEET_UNDER_MAX_OFFSET, 0.0, 1.0)
        )

        # ======================================================================
        # Cascading gates (smoothed over last _GATE_SMOOTHING steps)
        # ======================================================================
        # gate_stand: activated by feet reward
        gate_stand = self._gate_mean("feet")
        # gate_ws: activated by stand * gate_stand
        gate_ws = self._gate_mean("stand")
        # gate_march: activated by weight_shift * gate_ws
        gate_march = self._gate_mean("weight_shift")
        # gate_approach: activated by march * gate_march
        gate_approach = self._gate_mean("march")
        # gate_full: activated by approach * gate_approach
        gate_full = self._gate_mean("approach")

        # Stand-specific penalties fade out when the agent starts marching.
        stand_penalty_fade = 1.0 - gate_march

        # ======================================================================
        # Weight shift reward
        # ======================================================================
        com_to_feet = physics.com_lateral_to_foot()
        # Normalise by the *lateral* (y) foot separation, not the full xy
        # distance, so that spreading the feet in x cannot trivialise the
        # weight-shift reward.
        half_foot_dist = max(physics.feet_lateral_distance() / 2.0, 1e-6)
        shift_right = 1.0 - float(
            np.clip(np.abs(com_to_feet[0]) / half_foot_dist, 0.0, 1.0)
        )
        shift_left = 1.0 - float(
            np.clip(np.abs(com_to_feet[1]) / half_foot_dist, 0.0, 1.0)
        )

        current_side = "right" if shift_right > shift_left else "left"
        current_shift_value = max(shift_right, shift_left)

        if self._last_shift_side is not None and current_side != self._last_shift_side:
            if self._same_shift_count >= _WEIGHT_SHIFT_MIN_SAME:
                switch_bonus = _WEIGHT_SHIFT_SWITCH_BONUS
            else:
                switch_bonus = 0.0
            self._same_shift_count = 0
        else:
            switch_bonus = 0.0
            self._same_shift_count += 1
        self._last_shift_side = current_side

        if self._same_shift_count > _WEIGHT_SHIFT_MAX_SAME:
            ws_decay = max(
                0.0, 1.0 - (self._same_shift_count - _WEIGHT_SHIFT_MAX_SAME) / 20.0
            )
        else:
            ws_decay = 1.0

        weight_shift_reward = float(
            np.clip(current_shift_value * ws_decay + switch_bonus, 0.0, 1.0)
        )
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
        left_hip_lift = float(np.clip(hip_pitch[1] / _MARCH_HIP_PITCH_TARGET, 0.0, 1.0))

        right_lift = (right_knee_lift + right_hip_lift) / 2.0
        left_lift = (left_knee_lift + left_hip_lift) / 2.0

        touch_r = float(np.tanh(physics.data.sensordata[physics._sensor_r_foot]))
        touch_l = float(np.tanh(physics.data.sensordata[physics._sensor_l_foot]))
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

        if right_as_swing > left_as_swing:
            current_swing = "right"
            march_lift = float(right_as_swing)
        else:
            current_swing = "left"
            march_lift = float(left_as_swing)

        if self._last_swing_leg is not None and current_swing != self._last_swing_leg:
            if self._same_swing_count >= _MARCH_MIN_SAME:
                switch_bonus = _MARCH_SWITCH_BONUS
            else:
                switch_bonus = 0.0
            self._same_swing_count = 0
        else:
            switch_bonus = 0.0
            self._same_swing_count += 1
        self._last_swing_leg = current_swing

        if self._same_swing_count > _MARCH_MAX_SAME:
            march_decay = max(
                0.0, 1.0 - (self._same_swing_count - _MARCH_MAX_SAME) / 20.0
            )
        else:
            march_decay = 1.0

        march_stillness = 1.0 - float(
            np.clip(physics.horizontal_velocity() / _GAIT_MIN_VELOCITY, 0.0, 1.0)
        )

        march_reward = float(
            np.clip(
                (march_lift * march_decay + switch_bonus)
                * single_support
                * march_stillness,
                0.0,
                1.0,
            )
        )

        # ======================================================================
        # Approach + gait reward
        # ======================================================================
        torso_xy = physics.torso_xy()
        ball_xy = physics.ball_xy()
        dist_to_ball = np.linalg.norm(ball_xy - torso_xy)
        approach = float(
            rewards.tolerance(
                dist_to_ball,
                bounds=(0, _BALL_RADIUS),
                margin=3.0,
                value_at_margin=0.1,
                sigmoid="linear",
            )
        )
        if self._move_speed > 0:
            move_reward = float(
                rewards.tolerance(
                    physics.horizontal_velocity(),
                    bounds=(self._move_speed, float("inf")),
                    margin=self._move_speed / 2,
                    value_at_margin=0.5,
                    sigmoid="linear",
                )
            )
            approach_reward = approach * (5 * move_reward + 1) / 6
        else:
            approach_reward = approach

        h_vel = physics.horizontal_velocity()
        gait_gate = float(np.clip(h_vel / _GAIT_MIN_VELOCITY, 0.0, 1.0))
        # Foot clearance measured relative to the support foot (the lower one)
        # so that the reward reflects the actual *lift* of the swing foot,
        # independent of the robot's absolute height.  The target lift is one
        # shin length (_SHIN_LENGTH ≈ 0.25 m).
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
        # Reuse the single_support signal computed for march.
        gait_reward = gait_gate * (foot_clearance + single_support) / 2.0

        # ======================================================================
        # Kick + target reward
        # ======================================================================
        target_xy = physics.target_xy()
        ball_vel_xy = physics.ball_linear_velocity_xy()
        ball_to_target = target_xy - ball_xy
        ball_to_target_norm = np.linalg.norm(ball_to_target)
        if ball_to_target_norm > 1e-6:
            dir_to_target = ball_to_target / ball_to_target_norm
        else:
            dir_to_target = np.array([1.0, 0.0])
        ball_speed_toward = float(np.dot(ball_vel_xy, dir_to_target))
        kick_reward = float(np.tanh(ball_speed_toward / 5.0))  # [-1, 1]

        target_size = physics.get_target_size()
        dist_ball_to_target = np.linalg.norm(target_xy - ball_xy)
        target_hit = dist_ball_to_target < target_size + _BALL_RADIUS
        target_reward = 1.0 if target_hit else 0.0

        if target_hit:
            self._reset_ball_and_target(physics)

        # ======================================================================
        # Compute gated values for this step (to be stored in gate history)
        # ======================================================================
        feet_gated = feet_reward
        stand_gated = stand_reward * gate_stand
        ws_gated = weight_shift_reward * gate_ws * stillness
        march_gated = march_reward * gate_march
        approach_gated = approach_reward * gate_approach

        # Update gate history for the next step
        self._gate_history["feet"].append(feet_gated)
        self._gate_history["stand"].append(stand_gated)
        self._gate_history["weight_shift"].append(ws_gated)
        self._gate_history["march"].append(march_gated)
        self._gate_history["approach"].append(approach_gated)

        # ======================================================================
        # Final reward: all components, gated, single set of weights
        # ======================================================================
        reward = (
            # Positive rewards (sum of weights = 1.0)
            _W_FEET * feet_reward
            + _W_STAND * stand_gated
            + _W_SYMMETRY * symmetry_reward * gate_stand
            + _W_WEIGHT_SHIFT * ws_gated
            + _W_MARCH * march_gated
            + _W_APPROACH * approach_gated
            + _W_GAIT * gait_reward * gate_approach
            + _W_KICK * kick_reward * gate_full
            + _W_TARGET * target_reward * gate_full
            # Penalties (on top, small weights)
            + _W_EFFORT * effort_penalty
            + _W_FEET_UNDER * (feet_under - 1.0) * stand_penalty_fade
            + _W_HIP_ALIGN * hip_align_penalty * stand_penalty_fade
            + _W_LEG_SPREAD * leg_spread * stand_penalty_fade
            + _W_ANKLE_FLAT * (ankle_flat - 1.0) * stand_penalty_fade
            + _W_SMOOTHNESS * smoothness_penalty
        )

        # Log raw (unweighted) values for inspection
        self._reward_components = {
            "feet": feet_reward,
            "stand": stand_gated,
            "symmetry": symmetry_reward * gate_stand,
            "weight_shift": ws_gated,
            "march": march_gated,
            "approach": approach_gated,
            "gait": gait_reward * gate_approach,
            "kick": kick_reward * gate_full,
            "target": target_reward * gate_full,
            "effort": effort_penalty,
            "feet_under": (feet_under - 1.0) * stand_penalty_fade,
            "hip_align": hip_align_penalty * stand_penalty_fade,
            "leg_spread": leg_spread * stand_penalty_fade,
            "ankle_flat": (ankle_flat - 1.0) * stand_penalty_fade,
            "smoothness": smoothness_penalty,
            "gate_stand": gate_stand,
            "gate_ws": gate_ws,
            "gate_march": gate_march,
            "gate_approach": gate_approach,
            "gate_full": gate_full,
        }
        self._prev_action = ctrl.copy()
        return float(reward)
