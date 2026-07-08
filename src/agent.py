from src.learner import MPOLearner
from src.buffer import ReplayBuffer


class SoccerAgent:
    def __init__(self, learner: MPOLearner, buffer: ReplayBuffer, warmup: int, batch_size: int):
        self.learner = learner
        self.buffer = buffer
        self.state = learner.state

        self.batch_size = batch_size
        self.warmup = warmup * batch_size

    def train_step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.warmup:
            batch = self.buffer.next(self.batch_size)
            self.state, info = self.learner._update_step(self.state, batch)
            return info
        return None

    def select_action(self):
        # TODO
        raise NotImplementedError()
