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

## Development order
1. SDK client
2. Logger
3. GPS utils
4. Candidate planner
5. Hybrid controller
6. Command filter
7. Safety/recovery
8. Urban main loop

