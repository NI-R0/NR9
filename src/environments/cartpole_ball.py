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

"""Cartpole_ball domain."""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
import numpy as np


_DEFAULT_TIME_LIMIT = 15
SUITE = containers.TaggedTasks()
FILE = 'cartpole_ball.xml'


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_path = os.path.join(os.path.dirname(__file__), FILE)
  with open(xml_path, 'r') as f:
    xml_string = f.read()
  assets = {f"./common/{k}": v for k, v in common.ASSETS.items()}
  return xml_string, assets

@SUITE.add('benchmarking')
def kick(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns the kick task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Kick(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cartpole_ball domain."""

  def cart_position(self):
    """Returns the position of the cart."""
    return self.named.data.qpos['slider'][0]

  def angular_vel(self):
    """Returns the angular velocity of the pole."""
    return self.data.qvel[1:]

  def pole_angle_cosine(self):
    """Returns the cosine of the pole angle."""
    return self.named.data.xmat['lower_leg', 'zz']

  def ball_position(self):
    """Returns the [x, y, z] position of the ball."""
    return np.array(self.named.data.qpos['ball_joint'][:3])

  def ball_velocity(self):
    """Returns the [vx, vy, vz, wx, wy, wz] velocity of the ball."""
    return np.array(self.named.data.qvel['ball_joint'])


class Kick(base.Task):
  """A Cartpole_ball `Task` to kick the ball.

  State is initialized either close to the target configuration or at a random
  configuration.
  """

  def __init__(self, random=None):
    """Initializes an instance of `Kick`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.
    Args:
      physics: An instance of `Physics`.
    """
    physics.named.data.qpos['slider'] = -0.8 + .5*self.random.randn()
    physics.named.data.qpos['knee'] = np.pi + .01*self.random.randn()
    physics.named.data.qvel['slider'] = 0.01 * self.random.randn()
    physics.named.data.qvel['knee'] = 0.01 * self.random.randn()
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    obs['cart_position'] = np.array([physics.cart_position()])
    obs['cart_velocity'] = np.array([physics.named.data.qvel['slider'][0]])
    obs['pole_angle'] = np.array([physics.pole_angle_cosine(),
                                  physics.named.data.xmat['lower_leg', 'xz']])
    obs['pole_velocity'] = np.array([physics.named.data.qvel['knee'][0]])
    obs['ball_position'] = physics.ball_position()
    obs['ball_velocity'] = physics.ball_velocity()[:3]
    return obs

  def get_reward(self, physics):
    """Dense, bounded reward: approach ball + kick it far.

    All components are bounded to keep the reward scale stable:
      1. Ball x-velocity (primary: kick ball in +x), tanh-bounded to [-1, 1]
      2. Cart-to-ball proximity (shaping: reward getting close), [0, 1]
      3. Ball x-displacement from start (reward kicking far), tanh-bounded
    """
    ball_pos = physics.ball_position()
    ball_vel = physics.ball_velocity()
    cart_pos = physics.cart_position()

    ball_vel_x = float(ball_vel[0])
    vel_reward = np.tanh(ball_vel_x / 10.0)

    dist_to_ball = abs(float(ball_pos[0]) - 0.4 - cart_pos)
    proximity = np.exp(-2.0 * dist_to_ball)

    ball_displacement = max(0.0, float(ball_pos[0]) - 1.0)
    disp_reward = np.tanh(ball_displacement / 5.0)

    return float(vel_reward + 0.1 * proximity + disp_reward)
