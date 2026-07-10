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

"""one_joint_ball domain."""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
import numpy as np


_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()
FILE = 'one_joint_ball.xml'


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_path = os.path.join(os.path.dirname(__file__), FILE)
  with open(xml_path, 'r') as f:
    xml_string = f.read()
  # Map the common includes to the actual assets from dm_control
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
  """Physics simulation with additional features for the one_joint_ball domain."""

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

  def pole_tip_position(self):
    """Returns the [x, y, z] world position of the bottom tip of the pole.

    The lower_leg geom is a capsule from (0,0,0) to (0,0,1) in body-local
    coordinates, so the tip is at body_xpos + body_xmat @ [0, 0, 1].
    """
    body_pos = self.named.data.xpos['lower_leg']
    body_mat = self.named.data.xmat['lower_leg'].reshape(3, 3)
    tip_local = np.array([0.0, 0.0, 1.0])
    return body_pos + body_mat @ tip_local

class Kick(base.Task):
  """A one_joint_ball `Task` to kick the ball.

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
    physics.named.data.qpos['knee'] = np.pi + .01*self.random.randn()
    physics.named.data.qvel['knee'] = 0.01 * self.random.randn()
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    # Pole: angle (cos/sin) and angular velocity
    obs['pole_angle'] = np.array([physics.pole_angle_cosine(),
                                  physics.named.data.xmat['lower_leg', 'xz']])
    obs['pole_velocity'] = np.array([physics.named.data.qvel['knee'][0]])
    # Ball: position and velocity
    obs['ball_position'] = physics.ball_position()
    obs['ball_velocity'] = physics.ball_velocity()[:3]   # linear vx, vy, vz
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
    tip_pos = physics.pole_tip_position()

    # 1. Ball velocity in x-direction (main kick reward), bounded
    ball_vel_x = float(ball_vel[0])
    vel_reward = np.tanh(ball_vel_x / 10.0)  # bounded to [-1, 1]

    # 2. Proximity reward: closer pole tip -> higher reward
    dist_to_ball = np.linalg.norm(tip_pos[:2] - ball_pos[:2])
    proximity = np.exp(-2.0 * dist_to_ball)

    # 3. Ball displacement from initial position (x=1.0), bounded
    ball_displacement = max(0.0, float(ball_pos[0]) - 1.0)
    disp_reward = np.tanh(ball_displacement / 5.0)  # bounded to [0, 1)

    return float(vel_reward + 0.1 * proximity + disp_reward)
