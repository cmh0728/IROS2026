# Earth Rover Hybrid Autonomy

## Goal
Urban GPS MVP for Earth Rover Challenge using latency-aware hybrid reactive controller.

## Architecture
SDK -> Perception -> Candidate Planner -> Mode FSM -> Controller -> Command Filter -> SDK

## Setup
```bash
pip install -r requirements.txt
```

## Config
`configs/default.yaml`

## Run SDK smoke test
```bash
python scripts/run_sdk_smoke_test.py --config configs/default.yaml --no-motion
```

## Run Urban MVP
```bash
python scripts/run_urban_mvp.py --config configs/default.yaml
```

## Safety
Default motion limits are conservative. The system stops on stale frame, stale data, SDK failure, or emergency condition.

## SegFormer v2 Offline Planner Replay

`scripts/run_traversability_planner_replay_v2.sh` connects the approved SegFormer-B0 v2 checkpoint to the traversability adapter, goal-aware local planner, existing safety monitor, controller, and command filter. It reads recorded front-camera HLS data and writes expected commands to CSV/JSONL plus an H.264 review video.

This is a log-only integration gate. FrodoBots recordings do not provide the mission waypoint used by the Urban MVP, so the replay requires an explicit fixed heading error and records `gps_valid=false`, `goal_input_mode=fixed_heading_error`, and `command_transmitted=false`. It does not call the SDK or validate GPS navigation, recovery, or rover motion.

Run a five-second Dell smoke replay:

```bash
DURATION_SECONDS=5 RIDE_COUNT=1 LATENCY_SEC=0 GOAL_HEADING_ERROR_DEG=0 \
  ./scripts/run_traversability_planner_replay_v2.sh
```

Run the two-second delayed profile in a separate output directory:

```bash
LATENCY_SEC=2 GOAL_HEADING_ERROR_DEG=20 \
  ./scripts/run_traversability_planner_replay_v2.sh
```

The default output is
`$HOME/datasets/review_bundles/traversability_planner_replay_v2/latency_<N>s/`.
Each dataset directory contains `planner_replay.mp4`,
`logs/replay_steps.csv`, `logs/replay_steps.jsonl`, and
`review_manifest.json`. Generated data, checkpoints, and review videos remain
outside Git.

## Official SAM-TP Reproduction

SegFormer-B0 v2 remains the frozen lightweight semantic baseline. The separate
SAM-TP workflow reproduces only the official GeNIE perception model in an
independent Dell Conda or venv environment, applies a strict config/checkpoint gate, and
produces single-image logits, benchmark data, and deterministic FrodoBots
review videos. It does not train SAM-TP or connect it to the planner, SDK, or
live rover. See `docs/experiments/sam_tp_reproduction.md`.

## Development order
1. SDK client
2. Logger
3. GPS utils
4. Candidate planner
5. Hybrid controller
6. Command filter
7. Safety/recovery
8. Urban main loop
