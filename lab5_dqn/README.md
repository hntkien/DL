# Lab 5 — Value-Based Reinforcement Learning

Vanilla DQN, Double DQN, Prioritized Experience Replay, and *n*-step returns on `CartPole-v1` and `ALE/Pong-v5`. Spring 2026 Deep Learning (535518), NTU.

This submission directory is what the TA receives and runs from. All evaluation commands below assume **the submission root is the working directory** (i.e. the directory that contains this `README.md`).

## Directory layout

```
LAB5_R12942173_HuangChien/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── dqn.py                       # training script (all 3 tasks)
│   ├── test_cartpole.py             # Task 1 eval
│   ├── test_model.py                # Task 2 / Task 3 eval
│   ├── train_task{1,2,3}.sh         # training wrappers
│   ├── eval_task{1,2,3}.sh          # evaluation wrappers (used by TA)
│   └── configs/
│       ├── task1.yaml
│       ├── task2.yaml
│       ├── task3.yaml               # full enhanced: DDQN + PER + n-step=3
│       ├── task3_no_ddqn.yaml       # ablation
│       ├── task3_no_per.yaml        # ablation
│       └── task3_no_nstep.yaml      # ablation
├── LAB5_R12942173_task1.pt          # Task 1 best snapshot
├── LAB5_R12942173_task2.pt          # Task 2 best snapshot
├── LAB5_R12942173_task3_600000.pt   # Task 3 milestones
├── LAB5_R12942173_task3_1000000.pt
├── LAB5_R12942173_task3_1500000.pt
├── LAB5_R12942173_task3_2000000.pt
├── LAB5_R12942173_task3_2500000.pt
├── LAB5_R12942173_task3_best.pt     # any-step snapshot that reached score 19
├── LAB5_R12942173.mp4               # demo video
└── LAB5_R12942173.pdf               # report
```

The eval wrappers locate the `.pt` files via the glob pattern `LAB5_*_task<N>*.pt` at the project root, so no path needs to be edited for any specific student ID.

## Setup

Python 3.11 + PyTorch 2.5 with CUDA 12.4. CPU works but is slow for Pong.

```bash
pip install -r requirements.txt
```

(For W&B logging during training only: `wandb login`. Eval does not use W&B.)

## Evaluation (grader workflow)

All three commands run **20 episodes with seeds 0..19** and print per-episode reward followed by mean/std, matching the screenshot format in the grading PDF (Figure 4). Videos are skipped by default for speed; add `--with-video` to save mp4s under `videos/task<N>/`.

### Task 1 — CartPole-v1

```bash
bash code/eval_task1.sh
```

Equivalent direct call:
```bash
python code/test_cartpole.py --model-path LAB5_R12942173_task1.pt --episodes 20 --no-video
```

### Task 2 — ALE/Pong-v5 (vanilla DQN)

```bash
bash code/eval_task2.sh
```

Equivalent direct call:
```bash
python code/test_model.py --model-path LAB5_R12942173_task2.pt --episodes 20 --no-video
```

### Task 3 — ALE/Pong-v5 (enhanced DQN)

Evaluate **all six Task 3 snapshots** (5 milestones + best) in one command:

```bash
bash code/eval_task3.sh
```

The wrapper auto-discovers `LAB5_*_task3_*.pt` at the project root, sorts the numeric milestones ascending (600 k → 2.5 M), and appends `_best` last. Eval logs are written to `eval_logs/task3/eval_<snapshot_name>.txt`.

Evaluate a single Task 3 snapshot:

```bash
bash code/eval_task3.sh LAB5_R12942173_task3_2500000.pt
```

Equivalent direct call (note the two wrapper flags — they **must** be set for Task 3 snapshots; see "Notes on env wrappers" below):

```bash
python code/test_model.py \
    --model-path LAB5_R12942173_task3_2500000.pt \
    --episodes 20 --no-video \
    --disable-sticky-actions --pong-action-subset
```

## Reported results

| Task | Snapshot file | Mean over 20 eps | Notes |
|---|---|---|---|
| 1 | `LAB5_R12942173_task1.pt` | 500.00 ± 0.00 | seeds 0..19, CartPole-v1 |
| 2 | `LAB5_R12942173_task2.pt` | 19.70 | seeds 0..19, ALE/Pong-v5 |
| 3 best | `LAB5_R12942173_task3_best.pt` | 21.00 | first crossed 19 at ~880 k env steps |

Ablation summary (best-snapshot mean = 21.00 in all four runs; what differs is sample efficiency):

| Variant | Config | Env steps to reach 19 |
|---|---|---|
| Full (DDQN + PER + n=3) | `code/configs/task3.yaml` | ~880 k |
| − DDQN | `code/configs/task3_no_ddqn.yaml` | ~623 k |
| − PER | `code/configs/task3_no_per.yaml` | ~723 k |
| − *n*-step | `code/configs/task3_no_nstep.yaml` | ~623 k |

Full discussion is in the report.

## Reproducing training

Training writes checkpoints to `results/<task>/` (not into the submission root) and logs to W&B. Approximate wall-clock on a single RTX 3090: Task 1 ~5 min, Tasks 2/3 ~20 h each.

```bash
bash code/train_task1.sh                 # CartPole vanilla DQN
bash code/train_task2.sh                 # Pong vanilla DQN
bash code/train_task3.sh                 # Pong DDQN + PER + n-step
```

Any extra args override the YAML, for example:

```bash
bash code/train_task3.sh --episodes 1000 --wandb_run_name task3-smoke
```

Ablations re-use the same training script with a different config:

```bash
bash code/train_task3.sh --config code/configs/task3_no_ddqn.yaml
bash code/train_task3.sh --config code/configs/task3_no_per.yaml
bash code/train_task3.sh --config code/configs/task3_no_nstep.yaml
```

All Pong runs log to W&B project `DLP-Lab5-DQN-Pong`; CartPole runs log to `DLP-Lab5-DQN-CartPole`.