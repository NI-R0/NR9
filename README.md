# Learning to Score with a Robot ~~Soccer~~ Football Agent

Scoring a goal requires precise continuous control. The agent has to move toward the ball, make contact,
and kick it in the right direction.

Next steps:
- Get familiar with the DeepMind Control Soccer environment
- Simplify it if necessary
- Implement Maximum a Posteriori Policy Optimization (MPO)
- Train a robot agent to kick the ball (into the goal)

---

## Installation

Install `uv` and `python>=3.11` on your system and run:
```bash
git clone --recurse-submodules https://github.com/NI-R0/NR9.git
uv sync --frozen
```

If you forgot to add `--recurse-submodules` to the `git clone` command:
```bash
git submodule update --init
```

---

## Resources
- [MPO Paper](https://arxiv.org/pdf/1806.06920)
- [Official DeepMind MPO Implementation](https://github.com/google-deepmind/acme/tree/master/acme/agents/tf/mpo)
- [`dm_control` repository](https://github.com/google-deepmind/dm_control)