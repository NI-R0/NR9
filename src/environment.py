import numpy as np
from typing import Union
import src.environments.suite as suite
from dm_control import suite as dm_suite
from loguru import logger
import sys


class Environment:
    def __init__(
        self,
        domain_name: str = "cartpole",
        task_name: str = "balance",
        max_steps: int = 1000,
        use_icm: bool = False,
        icm_intrinsic_scale: float = 1.0,
        icm_lr: float = 5e-4,
        icm_hidden_sizes: tuple[int, ...] = (64, 32),
        icm_seed: int = 42,
    ):
        """Standard dm_control wrapper. Flattens dict observations into 1D arrays.

        Parameters
        ----------
        use_icm : bool
            If True, adds an Intrinsic Curiosity Module that computes a
            forward prediction error on flattened observations and adds
            it to the extrinsic reward.
        icm_intrinsic_scale : float
            Multiplicative factor applied to the prediction error.  Higher
            values amplify the curiosity signal.
        icm_lr : float
            Learning rate for the ICM forward model.
        icm_hidden_sizes : tuple[int, ...]
            Hidden layer sizes of the ICM MLP.
        icm_seed : int
            PRNG seed for ICM weight initialization.
        """
        self._preferred_camera: Union[str, int] = 0
        self.env = self._load_control_env(domain_name, task_name)

        self.action_spec = self.env.action_spec()
        self.action_dim = self.action_spec.shape

        first_timestep = self.env.reset()
        self.state_dim = self._flatten_observation(first_timestep.observation).shape

        self.ep_max_steps = max_steps
        self._setup_camera()

        # ── ICM ────────────────────────────────────────────────────────
        self._use_icm = use_icm
        self._icm = None
        if use_icm:
            from src.icm import ForwardModel
            self._icm = ForwardModel(
                obs_dim=int(np.prod(self.state_dim)),
                hidden_sizes=icm_hidden_sizes,
                lr=icm_lr,
                intrinsic_scale=icm_intrinsic_scale,
                seed=icm_seed,
            )
            logger.info(
                f"ICM enabled: scale={icm_intrinsic_scale}, "
                f"lr={icm_lr}, hidden={icm_hidden_sizes}"
            )
        self._last_state: np.ndarray | None = None

    def _setup_camera(self):
        """Pick the best available named camera, falling back to camera 0."""
        physics = self.env.physics

        def camera_exists(name: str) -> bool:
            try:
                physics.model.name2id(name, 'camera')
                return True
            except (KeyError, ValueError):
                return False

        for cam in ('side', 'back', 'lookatcart', 'fixed'):
            if camera_exists(cam):
                self._preferred_camera = cam
                return
        self._preferred_camera = 0

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
        state = self._flatten_observation(self.env.reset().observation)
        self._last_state = state.copy()
        return state

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_spec.minimum, self.action_spec.maximum)
        timestep = self.env.step(action)

        state = self._flatten_observation(timestep.observation)
        reward = timestep.reward if timestep.reward is not None else 0.0
        done = timestep.last()

        # Check custom early-termination (e.g. agent fell).
        # Only active if the task implements should_terminate and the
        # timestep is not already terminated by the time limit.
        if not done:
            task = getattr(self.env, 'task', None)
            if task is not None and hasattr(task, 'should_terminate'):
                if task.should_terminate(self.env.physics):
                    done = True

        # ── ICM intrinsic reward ──────────────────────────────────────
        icm_reward = 0.0
        if self._use_icm and self._icm is not None and self._last_state is not None:
            icm_reward = self._icm.update(self._last_state, state)
        self._last_state = state.copy()
        reward = reward + icm_reward

        info = {}
        task = getattr(self.env, 'task', None)
        if task is not None and hasattr(task, '_reward_components'):
            info['reward_components'] = dict(task._reward_components)
        if self._use_icm:
            if 'reward_components' not in info:
                info['reward_components'] = {}
            info['reward_components']['icm_reward'] = icm_reward

        return state, reward, done, info

    def render(self, height: int = 240, width: int = 320, camera_id: Union[str, int] = None):
        """Returns the current frame as an (H, W, 3) uint8 RGB array."""
        cam = camera_id if camera_id is not None else self._preferred_camera
        return self.env.physics.render(height=height, width=width, camera_id=cam)
