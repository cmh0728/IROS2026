# SAM-TP Full Navigation Pipeline Plan

Last updated: 2026-07-29

## 1. Objective

Connect the reproduced SAM-TP perception model to the existing Earth Rover
hybrid autonomy stack through staged, testable safety gates:

```text
SDK front frame
-> SAM-TP continuous traversability score
-> image-space adapter
-> calibrated BEV cost map
-> GPS goal-aware local path planner
-> path follower
-> existing safety, controller, and command filter
-> explicit ControlSink
```

The end state is a low-speed Urban navigation pipeline, but this plan does not
authorize live rover motion. Every live-motion stage requires explicit
authorization for that task and session.

## 2. Verified Starting Point

Current repository behavior is the source of truth.

- The official SAM-TP repository is frozen at
  `728aee296cf44288356de683b1948f18b05917d6`.
- The verified checkpoint is `checkpoint_2.pt`, SHA-256
  `2607fd6049d37f17fe96132cf35459f7e0a895107632410637d812756e3f9adb`.
- Training and inference architecture fields match.
- Model and checkpoint state dictionaries match exactly: 303 keys, no missing,
  unexpected, or shape-mismatched keys, followed by strict loading.
- Dell CUDA single-image inference passed with PyTorch `2.7.1+cu118`.
- Dell FP32 model-only warm inference is approximately 91 ms per frame, or
  approximately 10.9 FPS before SDK acquisition, decoding, planning, and
  control overhead.
- SAM-TP currently produces logits, a sigmoid traversability score, review
  videos, provenance, and benchmark reports only.
- SAM-TP is not connected to a planner, SDK source, controller, or rover.
- SegFormer-B0 v2 remains the frozen semantic baseline and currently has the
  only implemented learned-perception planner replay.
- `TraversabilityAdapter` expects SegFormer source IDs `0/1/2/3`; it is not a
  valid SAM-TP adapter.
- `GoalAwareLocalPlanner` currently selects one of `LEFT/CENTER/RIGHT`; it does
  not generate a geometric local path.
- `OfflineTraversabilityPipeline` uses the existing safety monitor, controller,
  command filter, and `LogOnlyControlSink`.
- `run_urban_mvp.py` uses `DummyTraversabilityModel` and can transmit SDK
  commands. It must not be used as the first SAM-TP integration entry point.

## 3. Architectural Decisions

### 3.1 Preserve Separate Model Semantics

SAM-TP predicts continuous membership in the bottom-point-prompted traversable
region. SegFormer predicts semantic source classes:

```text
0 IGNORE
1 ON_ROAD
2 OFF_ROAD
3 OBSTACLE
```

Do not manufacture semantic classes from SAM-TP scores. Do not pass SAM-TP
scores to the existing class-ID adapter. Introduce a model-neutral continuous
traversability contract and explicit model-specific adapters.

### 3.2 Use a Two-Stage Planner Migration

First connect SAM-TP to the existing sector planner in offline mode. This
validates timestamps, score direction, uncertainty policy, controller behavior,
and logging with minimal architecture change.

Then add a true geometric local planner:

```text
continuous traversability
-> calibrated rover-frame cost map
-> bounded candidate paths
-> selected LocalPath
```

The sector planner remains a fallback and regression oracle during migration.

### 3.3 Keep Semantic Safety Evidence

SAM-TP cannot identify obstacle type or certify physical clearance. The initial
full planner should support, behind configuration:

- SAM-TP as continuous free-space evidence;
- SegFormer `OBSTACLE` as a conservative veto or comparison channel;
- a SegFormer-only fallback when SAM-TP is unavailable, stale, or invalid;
- stop when neither perception source is valid.

Fusion must remain explicit and independently logged. A disagreement must not
be silently averaged away.

### 3.4 Keep Control Authority Explicit

Perception and planning never call the SDK. All commands flow through:

```text
planner target
-> existing controller
-> existing CommandFilter
-> safety override
-> ControlSink
```

`LogOnlyControlSink` is the default. `SdkControlSink` must require an explicit
live-motion feature flag and per-task authorization.

### 3.5 Preserve the Independent SAM-TP Environment

Do not merge the SAM-TP environment into the existing SDK or SegFormer
environment merely for convenience. Offline integration may run the project
code from the SAM-TP venv. Before shadow mode, choose and document the smallest
reliable deployment boundary after measuring serialization overhead:

- one integrated autonomy process launched in the SAM-TP venv; or
- a local SAM-TP perception worker with a bounded latest-frame interface.

Never allow an unbounded frame queue. New frames replace stale unprocessed
frames.

## 4. Target Contracts

Add contracts only when their phase begins.

### ContinuousTraversabilityPrediction

```text
raw_logits
score_map                  # float32, [0, 1], original image geometry
source_timestamp
received_timestamp
inference_started_at
inference_finished_at
model_version
input_shape
output_shape
valid
reason
```

The sigmoid score is not called calibrated confidence.

### ImageSpaceTraversability

```text
left_score
center_score
right_score
left_non_traversable_ratio
center_non_traversable_ratio
right_non_traversable_ratio
near_non_traversable_ratio
uncertainty_ratio
bottom_connected_ratio
recommended_direction
stop_recommended
reason
source_timestamp
inference_latency_ms
```

### LocalCostMap

```text
cost                       # normalized [0, 1], high means undesirable
observed_mask
resolution_m_per_cell
origin_in_rover_frame
source_timestamp
perception_age_sec
model_versions
valid
reason
```

Unknown or out-of-view cells remain unknown and receive a conservative cost.

### LocalPath

```text
points_xy_m
curvature
length_m
clearance_m
traversability_cost
goal_alignment_cost
smoothness_cost
total_cost
source_timestamps
valid
rejection_reason
```

### PipelineStep

One record must preserve frame, sensor, inference, planning, command, and sink
timestamps plus `command_transmitted`.

## 5. Phase Plan

Each phase is blocked until its completion gate passes.

### Phase 0 - Freeze Inputs and Establish the Hard Set

**Outcome**

Create a machine-readable integration baseline without changing either model.

**Reuse**

- `docs/experiments/sam_tp_reproduction.md`
- `docs/experiments/segformer_b0_v2_frozen_baseline_20260726.md`
- SAM-TP `review_manifest.json` and per-frame statistics
- SegFormer v2 temporal review manifests

**Work**

1. Record exact SAM-TP and SegFormer checkpoint hashes, configs, environments,
   input selections, and measured latency.
2. Build a paired review manifest for identical frames from
   `output_rides_0/1/2`.
3. Label review events, not pixels, for:
   `FALSE_TRAVERSABLE`, `FALSE_NON_TRAVERSABLE`, `FLICKER`,
   `CURB`, `STAIR`, `OBSTACLE`, `PAVED`, `GRAVEL`, `GRASS`,
   `SHADOW`, and `GLARE`.
4. Record when SAM-TP and SegFormer disagree.

**Verification**

- Exact frame/timestamp equality across paired model outputs.
- No train/validation/test reuse is presented as an independent quality test.
- No model threshold is tuned on a final test set.

**Gate**

At least one reproducible hard-set report identifies accepted behavior and
unsafe false-traversable cases. Reproduction success alone is not this gate.

### Phase 1 - SAM-TP Continuous Score Adapter

**Outcome**

Convert a SAM-TP score map into conservative image-space sector evidence without
inventing semantic labels.

**Expected files**

- `src/earth_rover/perception/sam_tp_adapter.py`
- model-neutral additions under `src/earth_rover/core/types.py`
- SAM-TP adapter config in a new replay profile
- focused tests under `tests/`

**Behavior**

1. Validate score shape, dtype, range, and finite values.
2. Preserve original image geometry and timestamps.
3. Use configurable road-relevant ROI and `LEFT/CENTER/RIGHT` sectors.
4. Weight the lower near field more strongly.
5. Measure bottom-connected high-score area separately from disconnected
   islands.
6. Treat low score as non-traversability evidence, not semantic obstacle proof.
7. Treat score near the decision boundary as uncertainty.
8. Stop on invalid, stale, collapsed, or insufficiently connected output.
9. Keep operational thresholding outside `SamTpPredictor` and record every
   threshold in config and reports.

**Focused tests**

- all-high, all-low, mixed-sector, disconnected-island, and nonfinite maps;
- resolution independence;
- stale timestamp behavior;
- low-margin uncertainty;
- bottom connectivity;
- deterministic output.

**Gate**

Synthetic tests pass and hard-set sector summaries match human interpretation.
No SDK or controller connection is allowed yet.

### Phase 2 - SAM-TP Offline Sector Planner Replay

**Outcome**

Run:

```text
recorded HLS
-> SAM-TP
-> SAM-TP adapter
-> existing GoalAwareLocalPlanner
-> existing safety/controller/filter
-> LogOnlyControlSink
```

**Reuse**

- `SamTpPredictor`
- deterministic HLS selection and decoder
- `OfflineTraversabilityPipeline`
- `GoalAwareLocalPlanner`
- `HybridReactiveController`
- `CommandFilter`
- `EmergencyStopMonitor`
- H.264 review rendering

**Work**

1. Generalize the offline pipeline input so SegFormer and SAM-TP adapters can
   share downstream planning without sharing model semantics.
2. Add explicit `perception_source=sam_tp`.
3. Use fixed/synthetic heading error only when recorded GPS goal data is absent;
   log `gps_valid=false`.
4. Produce latency `0` and delayed `2s` replays.
5. Log score maps, sector evidence, planner reason, expected commands, and
   `command_transmitted=false`.
6. Add side-by-side SAM-TP/SegFormer comparison mode without fusing them.

**Verification**

- synthetic end-to-end replay;
- unchanged SegFormer replay regression;
- same selection under fixed seed;
- stale frame and invalid score stop;
- command range and slew-rate limits;
- delayed replay never holds an old command indefinitely;
- review MP4 is H.264/yuv420p/10 FPS.

**Gate**

Human review accepts expected sector choices and stop behavior on the hard set.
SAM-TP must not produce a less conservative command on known false-traversable
cases.

### Phase 3 - Camera Calibration and Rover Geometry

**Outcome**

Establish the measurements required for a real local path in rover coordinates.

**Required user/hardware inputs**

- camera intrinsics and distortion;
- camera height, pitch, yaw, and translation relative to rover base;
- rover width, footprint, wheelbase-equivalent geometry, and safety margin;
- minimum practical turning radius or measured skid-steer response;
- verified SDK command sign and range;
- representative flat-ground calibration captures.

**Work**

1. Add versioned calibration files without usernames or machine paths.
2. Implement image-to-ground projection only within its valid ground-plane ROI.
3. Mark sky, above-horizon, occluded, and out-of-projection pixels unknown.
4. Inflate non-traversable/unknown regions by rover footprint and safety margin.
5. Preserve projection provenance and calibration version.

**Focused tests**

- known pixel-to-ground fixtures;
- inverse projection consistency where defined;
- horizon rejection;
- footprint inflation;
- resolution scaling;
- invalid/missing calibration rejection.

**Gate**

Measured calibration targets project within an agreed physical error bound.
Without this gate, output remains an image-space direction, not a local path.

### Phase 4 - BEV Cost Map and Perception Fusion

**Outcome**

Build a bounded rover-frame cost map.

**Behavior**

1. Convert SAM-TP score to a configurable continuous traversal cost.
2. Penalize uncertainty, unobserved cells, stale perception, and disconnected
   regions.
3. Optionally project SegFormer `OBSTACLE` as a hard or near-hard veto.
4. Preserve ON_ROAD/OFF_ROAD policy as planner cost, not a relabeling step.
5. Reject disagreement patterns identified as unsafe in the hard set.
6. Do not fill unknown space using aggressive morphology.

**Verification**

- synthetic masks with curb, corridor, fork, blocked center, and unknown gap;
- SAM-TP-only, SegFormer-only, fused, and invalid-source cases;
- conservative response to disagreement;
- no cost outside `[0, 1]`;
- timestamps and latency remain visible.

**Gate**

Every known false-traversable hard case is blocked or forces a conservative
state. If this is not possible, gather data or revise perception before path
planning.

### Phase 5 - Geometric Local Path Planner

**Recommended first implementation**

Use a deterministic constant-curvature candidate path bank in rover
coordinates. This is smaller, auditable, and compatible with the existing
classical controller. Audit the frozen GeNIE offline planner before reusing any
code; do not copy it until interfaces, assumptions, and licensing are resolved.

**Outcome**

Generate and score bounded candidate paths:

```text
traversability and clearance
+ GPS goal alignment
+ path progress
- unknown-space cost
- curvature and direction-change cost
- previous-path change cost
```

**Expected files**

- `src/earth_rover/planning/local_cost_map.py`
- `src/earth_rover/planning/local_path_planner.py`
- additions to core path types
- planner config and focused tests

**Behavior**

1. Generate paths only within configured curvature, length, and field of view.
2. Sweep rover footprint along every candidate.
3. Reject paths crossing blocked or insufficiently observed cells.
4. Prefer the GPS goal only among safe candidates.
5. Apply path hysteresis to reduce left/right oscillation.
6. Return STOP when no candidate clears the safety gate.
7. Record every component of every candidate score.

**Focused tests**

- clear center corridor;
- left/right blockage;
- GPS preference with all paths safe;
- GPS preference overridden by safety;
- narrow corridor below rover width;
- unknown gap;
- hysteresis;
- no valid path;
- deterministic tie breaking.

**Gate**

Synthetic scenarios and recorded hard-set replays produce physically plausible
paths with zero ride motion and zero SDK control calls.

### Phase 6 - Path Follower and Existing Controller Integration

**Outcome**

Convert `LocalPath` into a semantic steering and speed target, then reuse the
existing controller and command filter.

**Behavior**

1. Use a simple, bounded follower such as lookahead/pure-pursuit style tracking;
   do not add an optimization framework initially.
2. Scale lookahead and speed conservatively with path curvature, clearance,
   perception age, and confidence.
3. Keep final SDK `linear/angular` conversion and clamp in the existing control
   modules.
4. Emergency stop always overrides path following.
5. Recovery remains disabled until separately validated; do not infer reverse
   safety from the front camera.

**Verification**

- straight, left, right, high-curvature, and stop paths;
- command clamp and slew-rate limits;
- stale perception and GPS stop;
- emergency override;
- deterministic command sequence from a fixed replay;
- no SDK import in offline tests.

**Gate**

Expected commands are smooth, bounded, and conservative in latency `0` and
`2s` recorded replay.

### Phase 7 - Full Offline and Delayed Replay

**Outcome**

Produce a reviewable complete pipeline:

```text
recorded frame/sensors
-> SAM-TP
-> BEV/cost map
-> GPS goal-aware LocalPath
-> follower/controller/safety
-> LogOnlyControlSink
-> logs and review video
```

**Required outputs**

- source/receive/inference/plan/command timestamps;
- frame and sensor ages;
- cost map and selected/rejected paths;
- GPS validity and goal-input mode;
- safety state and reason;
- expected `linear/angular`;
- `command_transmitted=false`;
- effective FPS and peak VRAM;
- H.264 review video.

**Verification**

- short smoke replay, then multiple unseen rides;
- latency `0` and `2s`;
- dropped/malformed/stale frame cases;
- deterministic rerun;
- SegFormer fallback and disagreement cases;
- raw dataset/checkpoint fingerprints unchanged;
- generated artifacts excluded from Git.

**Gate**

Offline review passes with no unsafe expected command on the approved hard set.
This does not validate closed-loop navigation.

### Phase 8 - SDK Read-Only Shadow Mode

**Outcome**

Use live SDK frames and telemetry while prohibiting command transmission.

**New runtime combination**

```text
LiveSdkSensorSource
+ LogOnlyControlSink
```

**Safety requirements**

1. Do not call `/control`, `/start-mission`, `/checkpoint-reached`, or any
   motion endpoint.
2. Use a dedicated shadow entry point; do not modify `run_urban_mvp.py` to
   silently switch modes.
3. Keep a queue depth of one and replace stale unprocessed frames.
4. Record SDK timestamp, local receive timestamp, inference end, and frame age.
5. Stop the expected command on stale image/data, communication loss, invalid
   GPS, low confidence, invalid projection, or no safe path.
6. Show `command_transmitted=false` in every record and status view.

**Verification**

- mocked SDK contract tests;
- no-motion endpoint audit;
- 5-minute, then 30-minute attended shadow run;
- acquisition/inference/end-to-end p50/p95/max;
- dropped and stale frame counts;
- effective planning FPS;
- memory stability;
- manual comparison of expected commands with observed scene and GPS goal.

**Gate**

Shadow mode sustains the required rate without stale backlog, SDK writes, or
unsafe expected commands. This gate still does not authorize motion.

### Phase 9 - No-Motion Hardware Gate

**Outcome**

Validate the deployment on the rover while preventing movement.

**Requirements**

- explicit authorization for this task;
- physical emergency stop available;
- wheels raised or motion physically prevented when practical;
- control sink remains log-only or enforces zero commands;
- attended test only;
- documented stop conditions and user interruption method.

**Gate**

Frame freshness, GPS heading, path orientation, command signs, and stop override
are correct on actual hardware. No nonzero motion command is transmitted.

### Phase 10 - Limited Live Motion

This phase is a separate task and is not authorized by this plan.

**Prerequisites**

- all earlier gates pass;
- explicit per-task and per-session authorization;
- licensing/deployment status resolved for SAM-TP;
- controlled test area and spotter;
- verified emergency stop;
- lowest practical command and shortest practical duration;
- no unattended execution.

**Progression**

1. straight, sub-second low command;
2. commanded stop;
3. gentle left/right tracking;
4. single short local path;
5. single waypoint in a controlled area;
6. latency and communication-loss stop tests;
7. only then consider multi-waypoint Urban trials.

Stop immediately on stale telemetry, communication loss, unexpected motion,
invalid perception, invalid GPS, user interruption, or disagreement with the
expected path.

## 6. Configuration Strategy

Add configuration incrementally rather than creating one speculative full
profile. Expected sections are:

```text
sam_tp_adapter
camera_calibration
rover_geometry
local_cost_map
local_path_planner
path_follower
perception_fusion
shadow_mode
```

All paths are CLI or environment overrides. Tracked YAML must not contain
usernames, credentials, Tailscale addresses, dataset paths, or checkpoint
paths.

Feature defaults:

```text
sam_tp_enabled: false
perception_fusion_enabled: false
geometric_path_planner_enabled: false
sdk_shadow_mode: false
live_control_enabled: false
```

Missing checkpoint, calibration, or valid perception must never fall back to
pretending the feature is available.

## 7. Verification Matrix

| Layer | Unit/Synthetic | Recorded Replay | Delayed Replay | SDK Shadow | Live Motion |
|---|---:|---:|---:|---:|---:|
| SAM-TP adapter | required | required | required | required | prerequisite |
| BEV projection | required | required | required | required | prerequisite |
| Cost map/fusion | required | required | required | required | prerequisite |
| Local path planner | required | required | required | required | prerequisite |
| Follower/controller | required | required | required | required | prerequisite |
| Safety override | required | required | required | required | required |
| Command transmission | forbidden | forbidden | forbidden | forbidden | explicit authorization |

Every verification report must state the exact command, summarized result,
artifacts, unverified items, and whether hardware or motion was involved.

## 8. Logging and Review Requirements

Each planning step must retain:

- model/checkpoint/config/calibration versions;
- frame source and timestamps;
- sensor and GPS timestamps;
- inference and end-to-end latency;
- score-map statistics and validity;
- projection validity and observed-space ratio;
- SAM-TP/SegFormer disagreement;
- every candidate path and score component;
- selected path and rejection reason;
- follower target;
- raw and filtered expected commands;
- safety state and override reason;
- control sink type;
- `command_transmitted`.

The review video should show:

```text
original image
SAM-TP score overlay
BEV cost map
candidate paths
selected path
GPS goal direction
expected command
frame age, confidence, and safety state
```

## 9. Recommended Implementation Order

1. Phase 0 paired hard set.
2. Phase 1 SAM-TP adapter.
3. Phase 2 offline sector replay.
4. Phase 3 calibration and rover geometry.
5. Phase 4 BEV cost map and explicit fusion.
6. Phase 5 geometric local path planner.
7. Phase 6 follower/controller integration.
8. Phase 7 full offline and delayed replay.
9. Phase 8 SDK read-only shadow mode.
10. Phase 9 no-motion hardware gate.
11. Phase 10 limited live motion only under new explicit authorization.

The smallest safe first coding task is Phase 0 plus the synthetic core of Phase
1. It introduces no SDK dependency, no path claim, and no command path.

## 10. Full Definition of Done

The full pipeline is complete only when:

1. SAM-TP semantics and hard-set failure modes are documented.
2. Continuous score adaptation is deterministic and conservative.
3. Camera calibration and rover geometry are measured and versioned.
4. A rover-frame cost map preserves unknown space.
5. Candidate paths account for footprint, clearance, goal alignment, and
   uncertainty.
6. The existing safety, controller, and command filter remain authoritative.
7. Offline and two-second delayed replay pass.
8. SDK shadow mode proves frame freshness and stable throughput.
9. Every non-live record has `command_transmitted=false`.
10. No live path is enabled by default.
11. No live-motion claim is made without an explicitly authorized hardware test.
12. SegFormer remains available as the frozen baseline/fallback.

## 11. Material Unknowns Before Geometric Planning

These must be measured or explicitly confirmed during Phase 3:

- camera intrinsic and extrinsic calibration;
- rover footprint and safety clearance;
- minimum controllable speed and practical turning response;
- acceptable path projection error;
- required control/planning rate;
- whether the competition environment permits the SAM-TP upstream license
  status;
- whether SegFormer should be a hard veto, fallback, or comparison-only source.

Do not resolve these values by guesswork.
