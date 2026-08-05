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
git clone https://github.com/NI-R0/NR9.git
uv sync --frozen
```

`cd` into the project folder and run (for instance):
```bash
uv run python main.py \
    --task train \
    --run_name humanoid_custom \
    --env_domain stand \
    --env_task stand \
    --episodes 50000 \
    --num_envs 24 \
    --eval_frequency 50 \
    --num_eval_episodes 8 \
    --batch_size 512 \
    --capacity 500000 \
    --duration 27 \
```
See `src/cli.py` for all CLI options.

---

## Resources
- [MPO Paper](https://arxiv.org/pdf/1806.06920)
- [Official DeepMind MPO Implementation](https://github.com/google-deepmind/acme/tree/master/acme/agents/tf/mpo)
- [`dm_control` repository](https://github.com/deepmind/dm_control)

---

## Live Viewer with Checkpoint Hot-Swap (`--live`)

Launch the interactive dm_control viewer (requires a display) with the
trained agent.  Weights can be hot-swapped at runtime — no restart needed.

### Local file hot-swap

When `--checkpoint_poll_interval > 0`, the local checkpoint file is
polled and the agent's weights are hot-swapped when a new checkpoint is
saved by training.

```bash
uv run python main.py --task test \
  --load_dir runs/run_20260730_102646 --checkpoint latest \
  --live --respawn \
  --env_domain walker_3D_ball --env_task run \
  --checkpoint_poll_interval 5
```

### Remote checkpoint hot-swap (`--stream`)

When training runs on a cluster, use `--task serve` there to expose the
latest checkpoint over HTTP, forward the port, and connect locally with
`--stream`.  The viewer keeps running continuously; the checkpoint is
polled and downloaded in a **background thread** so the simulation never
stalls.  Weights are swapped instantly when the download finishes.

No `--load_dir` is needed on the local side — the initial checkpoint is
fetched from the server at startup.

1. **On the cluster** — start the checkpoint server:

```bash
uv run python main.py --task serve \
  --load_dir runs/run_20260730_102646 --checkpoint latest \
  --serve_port 2324
```

2. **Forward the port** (SSH or VS Code port forwarding):

```bash
ssh -L 2324:localhost:2324 user@cluster
```

3. **Locally** — launch the viewer with `--stream` (no `--load_dir` needed):

```bash
uv run python main.py --task test \
  --live --respawn \
  --env_domain walker_3D_ball --env_task run \
  --stream http://localhost:2324 --checkpoint_poll_interval 5
```

`--stream` accepts a full URL (`http://localhost:2324`), `host:port`
(`localhost:2324`), or just a port (`2324`).  Use
`--checkpoint_poll_interval` to control how often the server is polled
(defaults to 5 s when `--stream` is set).  Polling and downloading happen
in a background thread — the simulation is never interrupted.

---

## Headless Checkpoint Server (`--task serve`)

Runs on the cluster and serves the latest checkpoint over HTTP so that a
local machine can hot-swap weights via `--stream`.  No browser needed.

| Flag | Default | Description |
|------|---------|-------------|
| `--serve_port PORT` | 2324 | TCP port for the checkpoint server |
| `--stream URL` | off | Local: poll a remote checkpoint server for new weights |
| `--checkpoint_poll_interval SECONDS` | 0 | Poll frequency (0 = disabled; defaults to 5 s with `--stream`) |
| `--respawn` | off | Auto-reset env on termination |
