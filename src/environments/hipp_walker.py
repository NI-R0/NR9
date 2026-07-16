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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""hipp_walker Domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
import os

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 0.7

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10
FILE = 'hipp_walker.xml'

# Touch sensor names for non-foot limbs (used for reward penalty).
_NON_FOOT_TOUCHES = (
    'right_thigh_touch',
    'left_thigh_touch',
    'right_shin_touch',
    'left_shin_touch',
)

# All touch sensor names, including feet (used for observations).
_ALL_TOUCHES = _NON_FOOT_TOUCHES + (
    'right_right_foot_touch',
    'left_right_foot_touch',
    'right_left_foot_touch',
    'left_left_foot_touch',
)

SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_path = os.path.join(os.path.dirname(__file__), FILE)
  with open(xml_path, 'r') as f:
    xml_string = f.read()
  # Map the common includes to the actual assets from dm_control
  assets = {f"./common/{k}": v for k, v in common.ASSETS.items()}
  return xml_string, assets


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Stand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Humanoid(move_speed=0, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Humanoid(move_speed=_WALK_SPEED, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Humanoid(move_speed=_RUN_SPEED, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add()
def run_pure_state(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Humanoid(move_speed=_RUN_SPEED, pure_state=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker domain."""

  def torso_upright(self):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return self.named.data.xmat['torso', 'zz']

  def head_height(self):
    """Returns the height of the torso center of mass."""
    return self.named.data.subtree_com['torso', 'z']

  def center_of_mass_position(self):
    """Returns position of the center-of-mass."""
    return self.named.data.subtree_com['torso'].copy()

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_subtreelinvel'].copy()

  def torso_vertical_orientation(self):
    """Returns the z-projection of the torso orientation matrix."""
    return self.named.data.xmat['torso', ['zx', 'zy', 'zz']]

  def joint_angles(self):
    """Returns the state without global orientation or position."""
    return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint.
  
  def touch_forces(self):
    """Returns touch forces of all limbs (including feet) as a 1-D array."""
    return np.array([
        np.tanh(self.named.data.sensordata[name].item()-3)
        for name in _ALL_TOUCHES
    ])


  def extremities(self):
    """Returns end effector positions in egocentric frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    positions = []
    for side in ('left_', 'right_'):
      for limb in ('foot',):
        torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
        positions.append(torso_to_limb.dot(torso_frame))
    return np.hstack(positions)


class Humanoid(base.Task):
  """A humanoid task."""

  def __init__(self, move_speed, pure_state, random=None):
    """Initializes an instance of `Humanoid`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      pure_state: A bool. Whether the observations consist of the pure MuJoCo
        state or includes some useful features thereof.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._move_speed = move_speed
    self._pure_state = pure_state
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.

    """
    # Find a collision-free random initial configuration.
    penetrating = True
    while penetrating:
      randomizers.randomize_limited_and_rotational_joints(physics, self.random)
      # Check for collisions.
      physics.after_reset()
      penetrating = physics.data.ncon > 0
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns either the pure state or a set of egocentric features."""
    obs = collections.OrderedDict()
    if self._pure_state:
      obs['position'] = physics.position()
      obs['velocity'] = physics.velocity()
    else:
      obs['joint_angles'] = physics.joint_angles()
      obs['extremities'] = physics.extremities()
      obs['torso_vertical'] = physics.torso_vertical_orientation()
      obs['com_velocity'] = physics.center_of_mass_velocity()
      obs['velocity'] = physics.velocity()
      obs['touches'] = physics.touch_forces()
    return obs

  def get_reward(self, physics):
     """Returns a reward to the agent."""
     # --- Positiver Reward: je höher die Hüfte/Torso, desto mehr ---
     height_reward = np.tanh(physics.head_height())  # subtree_com['torso', 'z']
     

     # --- Negativer Reward: Kontakt von nicht-Füßen mit dem Boden ---
     touch_penalty = sum(
         np.tanh(physics.named.data.sensordata[name].item())
         for name in _NON_FOOT_TOUCHES
     )
     
     # Kombinieren: height_reward (0..1) minus touch_penalty
     reward = height_reward - touch_penalty

     # --- Optional: bestehende small_control / move-Logik beibehalten ---
     small_control = rewards.tolerance(physics.control(), margin=1,
                                       value_at_margin=0,
                                       sigmoid='quadratic').mean()
     small_control = (4 + small_control) / 5

     if self._move_speed == 0:
         horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
         dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
         reward = reward * small_control * dont_move
     else:
         com_velocity = np.linalg.norm(
             physics.center_of_mass_velocity()[[0, 1]])
         move = rewards.tolerance(com_velocity,
                                  bounds=(self._move_speed, float('inf')),
                                  margin=self._move_speed, value_at_margin=0,
                                  sigmoid='linear')
         move = (5 * move + 1) / 6
         reward = reward * small_control * move

     return reward