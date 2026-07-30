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
- [`dm_control` repository](https://github.com/deepmind/dm_control)

---

## Live Stream (Headless Cluster)

On a headless cluster (no display), use `--live-stream PORT` to serve an
MJPEG video stream over HTTP.  This works with offscreen rendering
(`MUJOCO_GL=egl`) and can be viewed in a local browser via SSH port
forwarding.

### Quick Start

1. **On the cluster** — start the stream (e.g. port 8080):

```bash
MUJOCO_GL=egl uv run python main.py -t test \
  --load_dir runs/run_20260730_102646 \
  --checkpoint latest \
  --live-stream 8080 \
  --respawn
```

2. **On your local machine** — forward the port via SSH:

```bash
ssh -L 8080:localhost:8080 user@cluster
```

3. Open `http://localhost:8080` in your browser.

### Checkpoint Hot-Swap

The stream watches the checkpoint file for changes.  When training
writes a new `latest.pkl` or `best_ckpt.pkl`, the agent's weights are
automatically reloaded without restarting the stream:

```bash
# Poll every 5 seconds (default: 10)
MUJOCO_GL=egl uv run python main.py -t test \
  --load_dir runs/run_20260730_102646 \
  --checkpoint latest \
  --live-stream 8080 \
  --checkpoint_poll_interval 5 \
  --respawn
```

Set `--checkpoint_poll_interval 0` to disable polling (load once at
startup).
