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

## Live Viewer with Checkpoint Hot-Swap (`--live`)

Launch the interactive dm_control viewer (requires a display) with the
trained agent.  When `--checkpoint_poll_interval > 0`, the checkpoint
file is polled and the agent's weights are hot-swapped when a new
checkpoint is saved by training — no restart needed.

```bash
uv run python main.py --task test \
  --load_dir runs/run_20260730_102646 --checkpoint latest \
  --live --respawn \
  --env_domain walker_3D_ball --env_task run \
  --checkpoint_poll_interval 5
```

Set `--checkpoint_poll_interval 0` (default) to disable hot-swap and
load the checkpoint once at startup.

---

## Headless MJPEG Stream (`--task serve`)

On a headless cluster (no display), use `--task serve` to run the agent
and stream rendered frames over HTTP as MJPEG.  Works with offscreen
rendering (`MUJOCO_GL=egl`) and includes checkpoint hot-swap.

### Quick Start

1. **On the cluster** — start the stream (e.g. port 8080):

```bash
MUJOCO_GL=egl uv run python main.py --task serve \
  --load_dir runs/run_20260730_102646 --checkpoint latest \
  --serve_port 8080 --checkpoint_poll_interval 5 --respawn \
  --env_domain walker_3D_ball --env_task run
```

2. **Forward the port** (SSH or VS Code port forwarding):

```bash
ssh -L 8080:localhost:8080 user@cluster
```

3. Open `http://localhost:8080` in your browser.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--serve_port PORT` | 2324 | TCP port for the HTTP MJPEG stream |
| `--checkpoint_poll_interval SECONDS` | 0 | Poll checkpoint for changes (0 = disabled) |
| `--respawn` | off | Auto-reset env on termination |
