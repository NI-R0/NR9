import numpy as np
import src.environments.suite as suite
from dm_control import suite as dm_suite
from loguru import logger
import sys


class Environment:
    def __init__(self, domain_name="cartpole", task_name="balance"):
        """Standard dm_control wrapper. Flattens dict observations into 1D arrays."""
        self.env = None
        try:
            self.env = dm_suite.load(domain_name=domain_name, task_name=task_name)
        except ValueError:
            self.env = suite.load(domain_name=domain_name, task_name=task_name)
        except Exception as e:
            logger.error(f"Could not load environment {domain_name} with task {task_name}: {e}")
            sys.exit(1)

        self.action_spec = self.env.action_spec()
        self.action_dim = self.action_spec.shape[0]

        # Run a dummy reset to determine the flattened state dimension
        dummy_timestep = self.env.reset()
        self.state_dim = self._flatten_obs(dummy_timestep.observation).shape[0]

        # dm_control defaults to 1000 steps per episode
        self.ep_max_steps = 1000

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        # Concatenates position, velocity, etc. into a single flat vector for the MLPs
        return np.concatenate([np.asarray(val).ravel() for val in obs_dict.values()])

    def reset(self) -> np.ndarray:
        timestep = self.env.reset()
        return self._flatten_obs(timestep.observation)

    def step(self, action: np.ndarray):
        # Clip action to physics bounds to prevent MuJoCo integration crashes
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)
        timestep = self.env.step(action)

        state = self._flatten_obs(timestep.observation)
        reward = timestep.reward if timestep.reward is not None else 0.0

        # dm_control indicates end of episode with last()
        done = timestep.last()

        return state, reward, done, {}

    def render(self):
        # In a real setup, you would grab pixels via self.env.physics.render()
        # and display them with cv2.imshow() or matplotlib.
        pixels = self.env.physics.render(camera_id=0)
        return pixels
