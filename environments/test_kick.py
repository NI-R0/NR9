import numpy as np
import environments.suite as suite
from dm_control import viewer

# CartPole "swingup" oder "balance" laden
env = suite.load(domain_name="cartpole_ball", task_name="kick")
# env = suite.load(domain_name="one_joint_ball", task_name="kick")

# Einfacher Random-Agent
class RandomAgent:
  def __init__(self, action_spec):
    self._action_spec = action_spec

  def __call__(self, time_step):
    return np.random.uniform(
        self._action_spec.minimum,
        self._action_spec.maximum,
        self._action_spec.shape)

agent = RandomAgent(env.action_spec())

# Environment im Viewer mit dem Agenten starten
viewer.launch(environment_loader=env, policy=agent)