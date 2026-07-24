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

Multi-stage reward (additive curriculum):
  0. Feet on ground only (penalise non-foot contact)
  1. Stand upright (height + orientation + hip alignment + symmetry + smoothness)
  2. Weight shift (shift centre of mass left/right while standing still)
  3. March in place (alternating knee lifts, single support, no forward motion)
  4. Approach the ball (walking toward it while upright, with gait quality)
  5. Kick the ball toward the target (ball velocity in target direction)
  6. Target hit: large bonus, then ball and target are randomly re-placed

Curriculum: the target zone shrinks after enough successful hits during
evaluation (success counter incremented externally via ``register_success``).
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
_CONTROL_TIMESTEP = .025
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

# Reward weights (additive across phases, not normalised)
_W_FEET = 0.2
_W_EFFORT = 0.05
_W_STAND = 0.25
_W_APPROACH = 0.3
_W_KICK = 0.2
_TARGET_BONUS = 100.0
_W_LEG_SPREAD = 0.1
_LEG_SPREAD_THRESHOLD = 0.2
_W_HIP_ALIGN = 0.1          # Hip-yaw/roll alignment penalty weight
_W_GAIT = 0.15              # Gait quality reward weight (foot clearance + single support)
_GAIT_MIN_VELOCITY = 0.3    # Min horizontal velocity (m/s) for gait reward to activate
_FOOT_CLEARANCE_TARGET = 0.12  # Target foot lift height (m) for gait reward
_HIP_YAW_MAX = np.radians(45)   # Max hip-yaw range for normalization
_HIP_ROLL_MAX = np.radians(45)  # Max hip-roll deviation from neutral for normalization
_W_SMOOTHNESS = 0.05        # Action smoothness penalty weight (anti-jitter)
_W_SYMMETRY = 0.1           # Mirror symmetry reward weight (left/right joint match)
_W_WEIGHT_SHIFT = 0.2       # Weight-shift reward weight (phase 2)
_W_MARCH = 0.25             # March-in-place reward weight (phase 3)
_MARCH_KNEE_TARGET = np.radians(60)  # Target knee lift angle for marching
_MARCH_HIP_PITCH_TARGET = np.radians(45)  # Target hip pitch for knee lift
_SYMMETRY_JOINT_MAX = np.radians(45)  # Normalization for symmetry reward
# Alternation parameters (march + weight_shift)
_MARCH_SWITCH_BONUS = 0.1      # Bonus for switching swing leg (small, just a nudge)
_MARCH_MIN_SAME = 15           # Min steps on same leg before a switch bonus is given
_MARCH_MAX_SAME = 40           # Steps before same-leg reward starts decaying (~1s)
_WEIGHT_SHIFT_SWITCH_BONUS = 0.1
_WEIGHT_SHIFT_MIN_SAME = 15
_WEIGHT_SHIFT_MAX_SAME = 40

# Touch sensor names for feet reward
_NON_FOOT_TOUCHES = (
    'torso_touch',
    'right_thigh_touch',
    'left_thigh_touch',
    'right_leg_touch',
    'left_leg_touch',
)
_FOOT_TOUCHES = (
    'right_foot_touch',
    'left_foot_touch',
)
_ALL_TOUCHES = _NON_FOOT_TOUCHES + _FOOT_TOUCHES

# Curriculum phases
PHASE_FEET = 0
PHASE_STAND = 1
PHASE_WEIGHT_SHIFT = 2
PHASE_MARCH = 3
PHASE_APPROACH = 4
PHASE_FULL = 5
_NUM_PHASES = 6

SUITE = containers.TaggedTasks()
FILE = 'walker_3D_ball.xml'


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_path = os.path.join(os.path.dirname(__file__), FILE)
  with open(xml_path, 'r') as f:
    xml_string = f.read()
  assets = {f"./common/{k}": v for k, v in common.ASSETS.items()}
  return xml_string, assets


def _make_task(move_speed, time_limit, random, environment_kwargs):
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Walker3DBall(move_speed=move_speed, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand+Kick task (move_speed=0 → focus on standing first)."""
  return _make_task(0, time_limit, random, environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk+Kick task."""
  return _make_task(_WALK_SPEED, time_limit, random, environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run+Kick task."""
  return _make_task(_RUN_SPEED, time_limit, random, environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker_3D_Ball domain."""

  def torso_upright(self):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return self.named.data.xmat['torso', 'zz']

  def torso_height(self):
    """Returns the world-z height of the top end of the torso capsule.

    The torso geom is a capsule with half-length 0.3 along the body's
    local z-axis.  The actual top position in world coordinates depends
    on the body orientation, so we project the half-length onto the
    world z-axis via ``xmat['torso', 'zz']`` (the z-component of the
    local z-axis in world frame).
    """
    center_z = self.named.data.xpos['torso', 'z']
    local_z_in_world = self.named.data.xmat['torso', 'zz']
    return center_z + 0.3 * local_z_in_world

  def torso_xy(self):
    """Returns the [x, y] position of the torso."""
    return np.array(self.named.data.xpos['torso'][:2])

  def horizontal_velocity(self):
    """Returns the horizontal speed of the center-of-mass (xy-plane)."""
    linvel = self.named.data.sensordata['torso_subtreelinvel']
    return np.linalg.norm(linvel[:2])

  def orientations(self):
    """Returns planar orientations of all bodies.

    For the 3D walker the full rotation matrix carries more information, but
    to keep the observation dimension manageable we return the same projection
    components (xx, xz) used by the planar walker for every non-torso body.
    """
    return self.named.data.xmat[1:, ['xx', 'xz']].ravel()

  def touch_forces(self):
    """Returns touch forces of all body parts (including feet) as a 1-D array.

    Each force is passed through ``tanh(force)`` so that light grazes
    produce ~0 while firm contacts saturate to ~1.  Consistent with
    ``feet_touch`` and ``non_foot_touch`` (no ``-3`` offset).
    """
    return np.array([
        np.tanh(self.named.data.sensordata[name].item())
        for name in _ALL_TOUCHES
    ])

  def feet_touch(self):
    """Returns summed tanh-saturated touch force for feet only."""
    return sum(
        np.tanh(self.named.data.sensordata[name].item())
        for name in _FOOT_TOUCHES
    )

  def non_foot_touch(self):
    """Returns summed tanh-saturated touch force for non-foot body parts."""
    return sum(
        np.tanh(self.named.data.sensordata[name].item())
        for name in _NON_FOOT_TOUCHES
    )

  def feet_horizontal_distance(self):
    """Returns the xy-distance between the two feet."""
    right_foot = self.named.data.xpos['right_foot'][:2]
    left_foot = self.named.data.xpos['left_foot'][:2]
    return float(np.linalg.norm(right_foot - left_foot))

  def ball_position(self):
    """Returns the [x, y, z] position of the ball."""
    return np.array(self.named.data.qpos['ball_joint'][:3])

  def ball_velocity(self):
    """Returns the [vx, vy, vz, wx, wy, wz] velocity of the ball."""
    return np.array(self.named.data.qvel['ball_joint'])

  def ball_xy(self):
    """Returns the [x, y] position of the ball."""
    return np.array(self.named.data.qpos['ball_joint'][:2])

  def ball_linear_velocity_xy(self):
    """Returns the [vx, vy] linear velocity of the ball."""
    return np.array(self.named.data.qvel['ball_joint'][:2])

  def target_position(self):
    """Returns the [x, y, z] world position of the target mocap body."""
    return np.array(self.named.data.xpos['target'])

  def target_xy(self):
    """Returns the [x, y] world position of the target."""
    return np.array(self.named.data.xpos['target'][:2])

  def set_target_position(self, xy):
    """Moves the mocap target body to ``(x, y, 0.1)`` and recomputes kinematics.

    ``physics.forward()`` is needed because ``data.xpos`` for mocap bodies is
    only updated during kinematics, which runs inside ``physics.step()`` - not
    when ``mocap_pos`` is set directly.
    """
    pos = np.array([xy[0], xy[1], 0.1], dtype=np.float64)
    self.named.data.mocap_pos['target'] = pos
    self.forward()

  def set_target_size(self, half_size):
    """Sets the target zone geom half-size in xy (box)."""
    self.named.model.geom_size['target_zone'] = [half_size, half_size, 0.05]

  def get_target_size(self):
    """Returns the current target zone half-size (xy)."""
    return float(self.named.model.geom_size['target_zone', 0])

  def hip_yaw_angles(self):
    """Returns [right_hip_yaw, left_hip_yaw] joint angles in radians."""
    return np.array([
        self.named.data.qpos['right_hip_yaw'],
        self.named.data.qpos['left_hip_yaw'],
    ]).ravel()

  def hip_roll_angles(self):
    """Returns [right_hip_roll, left_hip_roll] joint angles in radians."""
    return np.array([
        self.named.data.qpos['right_hip_roll'],
        self.named.data.qpos['left_hip_roll'],
    ]).ravel()

  def foot_heights(self):
    """Returns [right_foot_z, left_foot_z] world-z heights of the feet."""
    return np.array([
        self.named.data.xpos['right_foot'][2],
        self.named.data.xpos['left_foot'][2],
    ])

  def joint_positions(self):
    """Returns all non-root joint angles as a 1-D array (10 joints).

    Order: right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee,
    right_ankle, left_hip_yaw, left_hip_roll, left_hip_pitch,
    left_knee, left_ankle.
    """
    joint_names = [
        'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
        'right_knee', 'right_ankle',
        'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
        'left_knee', 'left_ankle',
    ]
    return np.array([self.named.data.qpos[name] for name in joint_names]).ravel()

  def knee_angles(self):
    """Returns [right_knee, left_knee] joint angles in radians."""
    return np.array([
        self.named.data.qpos['right_knee'],
        self.named.data.qpos['left_knee'],
    ]).ravel()

  def hip_pitch_angles(self):
    """Returns [right_hip_pitch, left_hip_pitch] joint angles in radians."""
    return np.array([
        self.named.data.qpos['right_hip_pitch'],
        self.named.data.qpos['left_hip_pitch'],
    ]).ravel()

  def com_lateral_offset(self):
    """Returns the lateral (y) distance from COM to the midpoint of the feet.

    Positive = COM shifted toward the left foot, negative = toward the right.
    Used for the weight-shift reward: the agent should shift its COM over
    one foot, then the other.
    """
    com_y = self.named.data.subtree_com['torso'][1]
    right_foot_y = self.named.data.xpos['right_foot'][1]
    left_foot_y = self.named.data.xpos['left_foot'][1]
    feet_mid_y = (right_foot_y + left_foot_y) / 2.0
    return float(com_y - feet_mid_y)

  def com_lateral_to_foot(self):
    """Returns [com_to_right_foot_y, com_to_left_foot_y] distances.

    A small absolute value means the COM is directly above that foot.
    """
    com_y = self.named.data.subtree_com['torso'][1]
    right_foot_y = self.named.data.xpos['right_foot'][1]
    left_foot_y = self.named.data.xpos['left_foot'][1]
    return np.array([com_y - right_foot_y, com_y - left_foot_y])


class Walker3DBall(base.Task):
  """3D walker with a multi-stage ball-kick reward and target curriculum.

  The reward progresses through stages (additive curriculum):
    0. *Feet* - reward foot-ground contact, penalise non-foot contact.
    1. *Stand* - torso height, upright orientation, hip alignment, symmetry,
       smoothness (added on top).
    2. *Weight shift* - shift COM over one foot then the other, stay still.
    3. *March* - alternating knee lifts in place, single support.
    4. *Approach* - walk toward the ball while upright, with gait quality.
    5. *Kick* - ball velocity in the direction of the target.
    6. *Target hit* - large bonus when the ball enters the target zone; ball
       and target are then randomly re-placed so the episode continues.

  Curriculum: ``register_success`` increments a counter.  After
  ``_SUCCESS_THRESHOLD`` consecutive successes the target zone shrinks by
  ``_TARGET_SHRINK`` (down to ``_TARGET_SIZE_MIN``).  A failure resets the
  consecutive-success counter.
  """

  def __init__(self, move_speed, random=None, phase=PHASE_FEET):
    """Initializes an instance of `Walker3DBall`.

    Args:
      move_speed: A float. If zero, the stand reward dominates. Otherwise this
        specifies a target horizontal velocity for the approach phase.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a
        seed automatically (default).
      phase: Curriculum phase (0=feet, 1=stand, 2=weight_shift, 3=march,
        4=approach, 5=full). Controls which reward components are active.
    """
    self._move_speed = move_speed
    self._target_size = _TARGET_SIZE_MAX
    self._consecutive_successes = 0
    self._target_pos = None
    self._phase = phase
    self._reward_components: dict[str, float] = {}
    self._prev_action: np.ndarray | None = None  # for action smoothness penalty
    self._last_swing_leg: str | None = None  # 'right' or 'left' (for march alternation)
    self._same_swing_count: int = 0           # consecutive steps with same swing leg
    self._last_shift_side: str | None = None  # 'right' or 'left' (for weight-shift alternation)
    self._same_shift_count: int = 0           # consecutive steps with same shift side
    super().__init__(random=random)

  def register_success(self):
    """Call when the agent successfully hits the target during evaluation.

    After ``_SUCCESS_THRESHOLD`` consecutive successes the target zone
    shrinks.  This is intended to be called from the training/eval loop.
    """
    self._consecutive_successes += 1
    if (self._consecutive_successes >= _SUCCESS_THRESHOLD
            and self._target_size > _TARGET_SIZE_MIN):
      self._target_size = max(
          _TARGET_SIZE_MIN, self._target_size - _TARGET_SHRINK)
      self._consecutive_successes = 0

  def register_failure(self):
    """Reset the consecutive-success counter."""
    self._consecutive_successes = 0

  def set_phase(self, phase: int):
    """Set the curriculum phase (0=feet, 1=stand, 2=weight_shift, 3=march,
    4=approach, 5=full).

    Called externally from the training loop after evaluation thresholds
    are met.  Controls which reward components are active.
    """
    if phase < 0 or phase >= _NUM_PHASES:
      raise ValueError(f"Invalid phase {phase}; must be 0-{_NUM_PHASES - 1}.")
    self._phase = phase

  @property
  def phase(self) -> int:
    return self._phase

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Resets the free-floating root to the nominal upright pose, randomizes
    joints, places the ball at its start position, and randomly places the
    target.  Also applies the current curriculum target size.
    """
    physics.named.data.qpos['root'] = [0.0, 0.0, 1.3, 1.0, 0.0, 0.0, 0.0]
    physics.named.data.qvel['root'] = 0.0
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    physics.named.data.qpos['ball_joint'] = list(_BALL_START_POS) + [1, 0, 0, 0]
    physics.named.data.qvel['ball_joint'] = 0.0

    physics.set_target_size(self._target_size)
    self._place_target(physics)
    self._prev_action = None  # reset action history
    self._last_swing_leg = None
    self._same_swing_count = 0
    self._last_shift_side = None
    self._same_shift_count = 0

    super().initialize_episode(physics)

  def _place_target(self, physics):
    """Randomly places the target at a random angle and distance."""
    angle = self.random.uniform(0, 2 * np.pi)
    dist = self.random.uniform(_TARGET_MIN_DIST, _TARGET_MAX_DIST)
    self._target_pos = np.array([dist * np.cos(angle),
                                  dist * np.sin(angle)])
    physics.set_target_position(self._target_pos)

  def _reset_ball_and_target(self, physics):
    """Re-places ball and target after a successful hit (mid-episode)."""
    physics.named.data.qpos['ball_joint'] = list(_BALL_START_POS) + [1, 0, 0, 0]
    physics.named.data.qvel['ball_joint'] = 0.0
    self._place_target(physics)

  def get_observation(self, physics):
    """Returns an observation of body state, ball, target, touches, and joints."""
    obs = collections.OrderedDict()
    obs['orientations'] = physics.orientations()
    obs['height'] = physics.torso_height()
    obs['velocity'] = physics.velocity()
    obs['ball_position'] = physics.ball_position()
    obs['ball_velocity'] = physics.ball_velocity()[:3]
    obs['target_position'] = physics.target_xy()
    obs['touches'] = physics.touch_forces()
    obs['joint_positions'] = physics.joint_positions()
    return obs

  def get_reward(self, physics):
    """Multi-stage reward, additive across curriculum phases.

    Each phase adds new reward components on top of the previous ones
    (no re-weighting), so Q-value predictions stay approximately stable
    when advancing phases.

    Phase 0 (feet):          feet + control
    Phase 1 (stand):         feet + control + stand + hip_align + symmetry
                             + smoothness
    Phase 2 (weight_shift):  + weight_shift (shift COM left/right, stay still)
    Phase 3 (march):         + march (alternating knee lifts, single support)
    Phase 4 (approach):      + approach + gait (walk to ball with gait quality)
    Phase 5 (full):          + kick + target bonus
    """
    # --- Feet reward: reward foot-only stance, penalise any non-foot contact ---
    feet_touch = physics.feet_touch()
    non_foot_touch = physics.non_foot_touch()
    feet_only = 1.0 - np.tanh(non_foot_touch)  # 1 when only feet touch
    feet_reward = np.tanh(feet_touch) * feet_only - np.tanh(non_foot_touch)
    feet_reward = float(np.clip(feet_reward, -1.0, 1.0))

    # --- Effort penalty: quadratic per-motor cost ---
    ctrl = physics.control()
    effort_penalty = float(np.mean(ctrl ** 2))

    reward = _W_FEET * feet_reward - _W_EFFORT * effort_penalty
    components = {
        'feet': _W_FEET * feet_reward,
        'effort': -_W_EFFORT * effort_penalty,
    }

    if self._phase == PHASE_FEET:
      self._reward_components = components
      self._prev_action = ctrl.copy()
      return float(reward)

    # ===== Phase 1+: Stand + alignment + symmetry + smoothness =====
    standing = float(np.clip(physics.torso_height() / _STAND_HEIGHT, 0.0, 1.0))
    upright = (1 + physics.torso_upright()) / 2
    stand_reward = (3 * standing + upright) / 4  # in [0, 1]
    reward += _W_STAND * stand_reward * feet_only

    # Leg-spread penalty
    feet_dist = physics.feet_horizontal_distance()
    leg_spread = float(np.clip(
        (feet_dist - _LEG_SPREAD_THRESHOLD) / _LEG_SPREAD_THRESHOLD, 0.0, 1.0
    ))
    reward -= _W_LEG_SPREAD * leg_spread * standing * feet_only

    # Hip-alignment penalty (discourage twisted legs)
    hip_yaw = physics.hip_yaw_angles()
    hip_roll = physics.hip_roll_angles()
    hip_align_penalty = float(
        np.mean(np.abs(hip_yaw) / _HIP_YAW_MAX)
        + np.mean(np.abs(hip_roll) / _HIP_ROLL_MAX)
    ) / 2.0  # in [0, 1]
    reward -= _W_HIP_ALIGN * hip_align_penalty * standing * feet_only

    # --- Mirror symmetry reward: left/right joint angles should match ---
    # Compares corresponding left/right joint pairs. When standing still,
    # both legs should be in the same configuration.
    joints = physics.joint_positions()  # [r_yaw, r_roll, r_pitch, r_knee, r_ankle,
                                        #  l_yaw, l_roll, l_pitch, l_knee, l_ankle]
    right_joints = joints[:5]
    left_joints = joints[5:]
    # Note: hip_pitch and knee have sign conventions where left/right
    # should be equal for a symmetric stance (not negated).
    symmetry_err = float(np.mean(np.abs(right_joints - left_joints) / _SYMMETRY_JOINT_MAX))
    symmetry_reward = 1.0 - np.clip(symmetry_err, 0.0, 1.0)  # in [0, 1]
    reward += _W_SYMMETRY * symmetry_reward * standing * feet_only

    # --- Action smoothness penalty: discourage jittery actions ---
    # Penalises large frame-to-frame action changes. Only applies when
    # we have a previous action to compare against.
    if self._prev_action is not None:
      action_diff = float(np.mean((ctrl - self._prev_action) ** 2))
    else:
      action_diff = 0.0
    reward -= _W_SMOOTHNESS * action_diff * standing * feet_only

    components['stand'] = _W_STAND * stand_reward * feet_only
    components['leg_spread'] = -_W_LEG_SPREAD * leg_spread * standing * feet_only
    components['hip_align'] = -_W_HIP_ALIGN * hip_align_penalty * standing * feet_only
    components['symmetry'] = _W_SYMMETRY * symmetry_reward * standing * feet_only
    components['smoothness'] = -_W_SMOOTHNESS * action_diff * standing * feet_only

    if self._phase == PHASE_STAND:
      self._reward_components = components
      self._prev_action = ctrl.copy()
      return float(reward)

    # ===== Phase 2+: Weight shift (COM left/right while standing still) =====
    # Reward shifting the COM over one foot, then the other.  Alternation
    # is enforced via a switch bonus and a decay for staying on the same
    # side too long.
    com_to_feet = physics.com_lateral_to_foot()  # [com_to_r_foot, com_to_l_foot]
    half_foot_dist = max(feet_dist / 2.0, 1e-6)
    shift_right = 1.0 - np.clip(np.abs(com_to_feet[0]) / half_foot_dist, 0.0, 1.0)
    shift_left = 1.0 - np.clip(np.abs(com_to_feet[1]) / half_foot_dist, 0.0, 1.0)

    # Determine which side the COM is over
    current_side = 'right' if shift_right > shift_left else 'left'
    current_shift_value = max(shift_right, shift_left)

    # Alternation tracking: bonus for switching (only after min hold time),
    # decay for staying too long.  The min-hold gate prevents rapid
    # oscillation to farm switch bonuses.
    if self._last_shift_side is not None and current_side != self._last_shift_side:
      if self._same_shift_count >= _WEIGHT_SHIFT_MIN_SAME:
        switch_bonus = _WEIGHT_SHIFT_SWITCH_BONUS
      else:
        switch_bonus = 0.0  # switched too fast, no bonus
      self._same_shift_count = 0
    else:
      switch_bonus = 0.0
      self._same_shift_count += 1
    self._last_shift_side = current_side

    # Decay reward after staying on same side too long
    if self._same_shift_count > _WEIGHT_SHIFT_MAX_SAME:
      decay = max(0.0, 1.0 - (self._same_shift_count - _WEIGHT_SHIFT_MAX_SAME) / 20.0)
    else:
      decay = 1.0

    weight_shift_reward = float(current_shift_value * decay + switch_bonus)
    # Penalise forward motion during weight shift (should stay in place)
    stillness = 1.0 - float(np.clip(
        physics.horizontal_velocity() / _GAIT_MIN_VELOCITY, 0.0, 1.0
    ))
    reward += _W_WEIGHT_SHIFT * weight_shift_reward * standing * feet_only * stillness

    components['weight_shift'] = (
        _W_WEIGHT_SHIFT * weight_shift_reward * standing * feet_only * stillness
    )

    if self._phase == PHASE_WEIGHT_SHIFT:
      self._reward_components = components
      self._prev_action = ctrl.copy()
      return float(reward)

    # ===== Phase 3+: March in place (alternating knee lifts) =====
    # Reward lifting one knee high while the other leg supports, then
    # switch.  Alternation is enforced via a switch bonus and a decay
    # for keeping the same swing leg too long.
    knee = physics.knee_angles()  # [right_knee, left_knee] (negative = flexed)
    hip_pitch = physics.hip_pitch_angles()  # [right, left] (positive = hip flexed)
    foot_z = physics.foot_heights()  # [right_foot_z, left_foot_z]

    # Knee flexion: knee angle range is [-150, 0], more negative = more flexed
    right_knee_lift = float(np.clip(-knee[0] / _MARCH_KNEE_TARGET, 0.0, 1.0))
    left_knee_lift = float(np.clip(-knee[1] / _MARCH_KNEE_TARGET, 0.0, 1.0))
    # Hip pitch: positive = lifting leg forward (flexion)
    right_hip_lift = float(np.clip(hip_pitch[0] / _MARCH_HIP_PITCH_TARGET, 0.0, 1.0))
    left_hip_lift = float(np.clip(hip_pitch[1] / _MARCH_HIP_PITCH_TARGET, 0.0, 1.0))

    right_lift = (right_knee_lift + right_hip_lift) / 2.0
    left_lift = (left_knee_lift + left_hip_lift) / 2.0

    touch_r = float(np.tanh(physics.named.data.sensordata['right_foot_touch'].item()))
    touch_l = float(np.tanh(physics.named.data.sensordata['left_foot_touch'].item()))
    touch_sum = touch_r + touch_l
    single_support = rewards.tolerance(
        touch_sum, bounds=(0.7, 1.3),
        margin=1.0, value_at_margin=0.1, sigmoid='linear',
    )

    right_as_swing = right_lift * (1.0 - touch_r) * touch_l  # right up, left down
    left_as_swing = left_lift * (1.0 - touch_l) * touch_r  # left up, right down

    # Determine which leg is currently the swing leg
    if right_as_swing > left_as_swing:
      current_swing = 'right'
      march_lift = float(right_as_swing)
    else:
      current_swing = 'left'
      march_lift = float(left_as_swing)

    # Alternation tracking: bonus for switching (only after min hold time),
    # decay for staying too long.  The min-hold gate prevents rapid
    # oscillation to farm switch bonuses.
    if self._last_swing_leg is not None and current_swing != self._last_swing_leg:
      if self._same_swing_count >= _MARCH_MIN_SAME:
        switch_bonus = _MARCH_SWITCH_BONUS
      else:
        switch_bonus = 0.0  # switched too fast, no bonus
      self._same_swing_count = 0
    else:
      switch_bonus = 0.0
      self._same_swing_count += 1
    self._last_swing_leg = current_swing

    # Decay reward after keeping same swing leg too long
    if self._same_swing_count > _MARCH_MAX_SAME:
      decay = max(0.0, 1.0 - (self._same_swing_count - _MARCH_MAX_SAME) / 20.0)
    else:
      decay = 1.0

    march_stillness = 1.0 - float(np.clip(
        physics.horizontal_velocity() / _GAIT_MIN_VELOCITY, 0.0, 1.0
    ))

    march_reward = (march_lift * decay + switch_bonus) * single_support * march_stillness
    reward += _W_MARCH * march_reward * standing * feet_only

    components['march'] = _W_MARCH * march_reward * standing * feet_only

    if self._phase == PHASE_MARCH:
      self._reward_components = components
      self._prev_action = ctrl.copy()
      return float(reward)

    # ===== Phase 4+: Approach (walk to ball with gait quality) =====
    torso_xy = physics.torso_xy()
    ball_xy = physics.ball_xy()
    dist_to_ball = np.linalg.norm(ball_xy - torso_xy)
    approach = rewards.tolerance(dist_to_ball,
                                 bounds=(0, _BALL_RADIUS),
                                 margin=3.0,
                                 value_at_margin=0.1,
                                 sigmoid='linear')
    if self._move_speed > 0:
      move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                      bounds=(self._move_speed, float('inf')),
                                      margin=self._move_speed / 2,
                                      value_at_margin=0.5,
                                      sigmoid='linear')
      approach_reward = approach * (5 * move_reward + 1) / 6
    else:
      approach_reward = approach
    reward += _W_APPROACH * approach_reward

    # --- Gait quality reward: encourage proper walking, not shuffling ---
    h_vel = physics.horizontal_velocity()
    gait_gate = float(np.clip(h_vel / _GAIT_MIN_VELOCITY, 0.0, 1.0))

    swing_foot_z = float(np.max(foot_z))
    foot_clearance = rewards.tolerance(
        swing_foot_z, bounds=(_FOOT_CLEARANCE_TARGET, float('inf')),
        margin=_FOOT_CLEARANCE_TARGET, value_at_margin=0.1,
        sigmoid='linear',
    )

    # Recompute single_support (already computed above but may have been
    # overwritten by march; recompute for gait context)
    single_support_gait = rewards.tolerance(
        touch_sum, bounds=(0.7, 1.3),
        margin=1.0, value_at_margin=0.1, sigmoid='linear',
    )

    gait_reward = gait_gate * (foot_clearance + single_support_gait) / 2.0
    reward += _W_GAIT * gait_reward

    components['approach'] = _W_APPROACH * approach_reward
    components['gait'] = _W_GAIT * gait_reward

    if self._phase == PHASE_APPROACH:
      self._reward_components = components
      self._prev_action = ctrl.copy()
      return float(reward)

    # ===== Phase 5: Full (kick + target) =====
    target_xy = physics.target_xy()
    ball_vel_xy = physics.ball_linear_velocity_xy()
    ball_to_target = target_xy - ball_xy
    ball_to_target_norm = np.linalg.norm(ball_to_target)
    if ball_to_target_norm > 1e-6:
      dir_to_target = ball_to_target / ball_to_target_norm
    else:
      dir_to_target = np.array([1.0, 0.0])
    ball_speed_toward = float(np.dot(ball_vel_xy, dir_to_target))
    kick_reward = np.tanh(ball_speed_toward / 5.0)
    reward += _W_KICK * kick_reward

    target_size = physics.get_target_size()
    dist_ball_to_target = np.linalg.norm(target_xy - ball_xy)
    target_hit = dist_ball_to_target < target_size + _BALL_RADIUS
    if target_hit:
      reward += _TARGET_BONUS
      self._reset_ball_and_target(physics)

    components['kick'] = _W_KICK * kick_reward
    components['target'] = _TARGET_BONUS if target_hit else 0.0

    self._reward_components = components
    self._prev_action = ctrl.copy()
    return float(reward)
