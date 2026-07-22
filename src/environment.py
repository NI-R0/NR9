import numpy as np
import src.environments.suite as suite
from dm_control import suite as dm_suite
from loguru import logger
import sys


class Environment:
    def __init__(self, domain_name: str = "cartpole", task_name: str = "balance", max_steps: int = 1000):
        """Standard dm_control wrapper. Flattens dict observations into 1D arrays."""
        self.env = self._load_control_env(domain_name, task_name)

        self.action_spec = self.env.action_spec()
        self.action_dim = self.action_spec.shape

        first_timestep = self.env.reset()
        self.state_dim = self._flatten_observation(first_timestep.observation).shape

        self.ep_max_steps = max_steps

    def _load_control_env(self, domain_name: str, task_name: str):
        try:
            return dm_suite.load(domain_name=domain_name, task_name=task_name)
        except ValueError:
            pass
        try:
            return suite.load(domain_name=domain_name, task_name=task_name)
        except Exception as e:
            logger.error(f"Could not load environment {domain_name} with task {task_name}: {e}")
            sys.exit(1)

    def _flatten_observation(self, obs_dict: dict) -> np.ndarray:
        return np.concatenate([np.asarray(val).ravel() for val in obs_dict.values()]).astype(np.float32)

    def reset(self) -> np.ndarray:
        return self._flatten_observation(self.env.reset().observation)

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)
        timestep = self.env.step(action)

        state = self._flatten_observation(timestep.observation)
        reward = timestep.reward if timestep.reward is not None else 0.0
        done = timestep.last()

        return state, reward, done, {}

    def render(self, height: int = 240, width: int = 320, camera_id: int = 0):
        """Returns the current frame as an (H, W, 3) uint8 RGB array.

        Requires a configured MuJoCo GL backend (env var MUJOCO_GL=egl or
        osmesa for headless offscreen rendering, glfw if a real display is
        available), set before dm_control is imported.

        Author's Note: Rendering process designed by Claude. 
        """
        return self.env.physics.render(height=height, width=width, camera_id=camera_id)
