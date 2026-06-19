from dm_control import viewer
from dm_control.locomotion import soccer as dm_soccer
import numpy as np

env = dm_soccer.load(team_size=2,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

action_specs = env.action_spec()


def random_policy(time_step):
    actions = []
    for action_spec in action_specs:
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
    return actions


# Does not work on WSL or headless servers!
viewer.launch(env, policy=random_policy)
