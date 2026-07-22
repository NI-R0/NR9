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

Multi-stage reward:
  1. Stand upright (height + orientation)
  2. Approach the ball (walking toward it while upright)
  3. Kick the ball toward the target (ball velocity in target direction)
  4. Target hit: large bonus, then ball and target are randomly re-placed

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
_STAND_HEIGHT = 1.2
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
_W_STAND = 0.2
_W_APPROACH = 0.3
_W_KICK = 0.5
_TARGET_BONUS = 10.0


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
    """Returns the height of the torso."""
    return self.named.data.xpos['torso', 'z']

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


class Walker3DBall(base.Task):
  """3D walker with a multi-stage ball-kick reward and target curriculum.

  The reward progresses through stages:
    1. *Stand* - torso height and upright orientation (always active).
    2. *Approach* - reward for reducing torso-to-ball distance while upright.
    3. *Kick* - reward for ball velocity in the direction of the target.
    4. *Target hit* - large bonus when the ball enters the target zone; ball
       and target are then randomly re-placed so the episode continues.

  Curriculum: ``register_success`` increments a counter.  After
  ``_SUCCESS_THRESHOLD`` consecutive successes the target zone shrinks by
  ``_TARGET_SHRINK`` (down to ``_TARGET_SIZE_MIN``).  A failure resets the
  consecutive-success counter.
  """

  def __init__(self, move_speed, random=None):
    """Initializes an instance of `Walker3DBall`.

    Args:
      move_speed: A float. If zero, the stand reward dominates. Otherwise this
        specifies a target horizontal velocity for the approach phase.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a
        seed automatically (default).
    """
    self._move_speed = move_speed
    self._target_size = _TARGET_SIZE_MAX
    self._consecutive_successes = 0
    self._target_pos = None
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
    """Returns an observation of body state, ball, and target."""
    obs = collections.OrderedDict()
    obs['orientations'] = physics.orientations()
    obs['height'] = physics.torso_height()
    obs['velocity'] = physics.velocity()
    obs['ball_position'] = physics.ball_position()
    obs['ball_velocity'] = physics.ball_velocity()[:3]
    obs['target_position'] = physics.target_xy()
    return obs

  def get_reward(self, physics):
    """Multi-stage reward: stand + approach + kick + target bonus."""
    standing = rewards.tolerance(physics.torso_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT / 2)
    upright = (1 + physics.torso_upright()) / 2
    stand_reward = (3 * standing + upright) / 4

    torso_xy = physics.torso_xy()
    ball_xy = physics.ball_xy()
    target_xy = physics.target_xy()
    target_size = physics.get_target_size()
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

    ball_vel_xy = physics.ball_linear_velocity_xy()
    ball_to_target = target_xy - ball_xy
    ball_to_target_norm = np.linalg.norm(ball_to_target)
    if ball_to_target_norm > 1e-6:
      dir_to_target = ball_to_target / ball_to_target_norm
    else:
      dir_to_target = np.array([1.0, 0.0])
    ball_speed_toward = float(np.dot(ball_vel_xy, dir_to_target))
    kick_reward = np.tanh(ball_speed_toward / 5.0)

    dist_ball_to_target = np.linalg.norm(target_xy - ball_xy)
    target_bonus = 0.0
    if dist_ball_to_target < target_size + _BALL_RADIUS:
      target_bonus = _TARGET_BONUS
      self._reset_ball_and_target(physics)

    reward = (_W_STAND * stand_reward
              + _W_APPROACH * approach_reward
              + _W_KICK * kick_reward
              + target_bonus)
    return float(reward)
