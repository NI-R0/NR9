# Learning to Score with a Robot Soccer Agent

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

