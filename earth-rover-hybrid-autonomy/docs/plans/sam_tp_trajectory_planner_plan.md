# SAM-TP Image-Space Trajectory Planner Plan

Last updated: 2026-07-30

## 1. Objective and Scope

Connect the frozen official SAM-TP perception baseline to the existing Earth
Rover hybrid autonomy stack through staged offline and shadow-mode gates:

```text
front camera
-> SAM-TP continuous traversability score
-> constant-curvature trajectory candidates
-> image-space trajectory corridor scoring
-> GPS-aware local path selection
-> existing controller, command filter, safety, and recovery
-> offline/delayed replay
-> read-only SDK shadow mode
```

SAM-TP never emits `linear` or `angular` commands. The planner selects a local
path; the existing classical controller produces an expected command; the
existing safety layer may override it. This plan does not authorize SDK control
or live rover motion.

The first geometric planner intentionally scores projected rover corridors in
the original SAM-TP image-space score map. It does not build a full BEV cost
map. BEV remains a later extension point for depth, temporal metric mapping, or
demonstrated image-space limitations.

The following work is explicitly out of scope:

- SAM-TP or SegFormer training;
- additional labeling or model comparison;
- full BEV mapping, SLAM, or 3D reconstruction;
- A*, RRT*, or optimization-based planning;
- end-to-end learned control;
- SDK command transmission or live autonomous driving.

## 2. Verified Repository State

Repository inspection at plan creation found:

- branch `main`;
- HEAD `1b4df44` (`Support additional SAM-TP review datasets`);
- an unrelated user modification to the root `.gitignore`, which must remain
  untouched and excluded from planner commits;
- a separate `frodobots-earth-rover/` Git repository, which is not part of this
  implementation.

### Existing components to reuse

- Official SAM-TP lifecycle and validated output:
  `training/sam_tp_reproduction.py`
  - `SamTpPredictor`
  - `SamTpPrediction`
  - `sigmoid_logits`
  - score range, dtype, shape, and finite-value validation
- Deterministic HLS selection and decoding:
  - `training/run_sam_tp_video_review.py`
  - `training/run_traversability_video_review_v2.py`
  - `training/traversability_video_review_v2.py`
- Read-only SDK perception shadow:
  - `training/sam_tp_sdk_shadow.py`
  - `training/run_sam_tp_sdk_shadow.py`
- GPS and waypoint behavior:
  - `earth_rover.navigation.gps_utils`
  - `HeadingFilter`
  - `WaypointManager`
- Existing control:
  - `HybridReactiveController`
  - `CommandFilter`
  - `LatencyCompensator`
- Existing safety and recovery:
  - `EmergencyStopMonitor`
  - `StuckDetector`
  - `RecoveryController`
- Offline interfaces and logging:
  - `SensorSource`
  - `ControlSink`
  - `LogOnlyControlSink`
  - `OfflineTraversabilityPipeline`
- Recorded-run delayed replay:
  - `scripts/run_delayed_replay.py`
- Runtime logging:
  - `RunLogger`

### Components that must not be misused

`earth_rover.perception.traversability_adapter.TraversabilityAdapter` consumes
SegFormer source IDs `0/1/2/3`; it is not compatible with SAM-TP continuous
scores. `GoalAwareLocalPlanner` and `CandidateDirectionPlanner` select broad
LEFT/CENTER/RIGHT sectors and do not represent geometric trajectories. Their
weighting, safety-gating, hysteresis, and deterministic tie-breaking patterns
may be reused, but SAM-TP scores must not be converted into fake semantic
classes.

`scripts/run_urban_mvp.py` can transmit SDK commands. It must not be used as the
initial SAM-TP planner integration entry point.

## 3. SAM-TP Output Semantics

The frozen official wrapper returns an `HxW` raw-logit map in source-frame
geometry. The project computes:

```text
traversability_score = sigmoid(raw_logits)
```

Score semantics are:

```text
near 0 -> low traversability evidence
near 1 -> high traversability evidence
```

The score is not a calibrated probability and is not obstacle semantics.
Visualization heatmaps are not planner inputs. The current wrapper restores the
score to the source image resolution, so planner pixel coordinates refer to the
original frame. Any future preprocessing change must preserve or explicitly
record the inverse coordinate transform.

A scalar data-quality value must not be derived from `abs(score - 0.5)`.
Initially it is the valid-pixel fraction. Model ambiguity and temporal
instability are separate scorer/planner diagnostics.

## 4. Recommended Architecture

```text
Replay HLS / RunLogger timeline / ReadOnlySdkSource
-> existing SamTpPredictor
-> SamTpOutputAdapter
-> ConstantCurvatureTrajectorySampler
-> CameraTrajectoryProjector
-> ImageSpaceTrajectoryScorer
-> SamTpLocalPlanner
-> PlannerControllerAdapter
-> existing HybridReactiveController
-> existing CommandFilter
-> existing EmergencyStopMonitor / RecoveryController
-> LogOnlyControlSink
-> existing H.264 review writer
```

The coordinate conventions are:

```text
rover/base frame: +x forward, +y left, +z up
curvature: positive left, negative right
image frame: +u right, +v down
camera optical frame: +x right, +y down, +z forward
controller angular command: positive left
```

## 5. Interface Specification

### TraversabilityOutput

```python
@dataclass(frozen=True)
class TraversabilityOutput:
    score_map: np.ndarray
    valid_mask: np.ndarray
    confidence: float
    frame_timestamp: float
    inference_time_ms: float
    model_version: str
```

`score_map` is read-only float32 in `[0, 1]`. `valid_mask` is read-only bool and
has the same shape. `confidence` is the valid-pixel fraction. Invalid score
shape, range, timestamp, latency, non-finite value, or timeout raises a stable
adapter error that a later pipeline converts to a safety stop reason.

### CandidateTrajectory

```python
@dataclass(frozen=True)
class CandidateTrajectory:
    curvature: float
    points_xy: np.ndarray
    headings_rad: np.ndarray
    horizon_m: float
    sample_distances_m: np.ndarray
    left_boundary_xy: np.ndarray
    right_boundary_xy: np.ndarray
    effective_half_width_m: float
```

The sampled distance is path arc length. It begins at zero and ends exactly at
the configured horizon. The footprint half-width is:

```text
rover_width_m / 2 + safety_margin_m
```

### ProjectedTrajectory

Planned interface:

```text
trajectory
centerline_uv
left_boundary_uv
right_boundary_uv
corridor_mask
near/mid/far masks
valid cross-section ratio
projection_valid
reason
```

Projection rejects camera-behind, non-positive depth, horizon-invalid, and
out-of-image geometry. It does not invent visible ground outside the camera
field of view.

### TrajectoryScore

Planned interface:

```text
total score
weighted mean traversability
lower-quantile traversability
near-field risk
unknown ratio
valid ratio
goal alignment
curvature/change costs
hard rejection state and reason
```

### PlannerOutput

```python
@dataclass(frozen=True)
class PlannerOutput:
    selected_curvature: float
    local_target_xy: tuple[float, float]
    path_score: float
    path_confidence: float
    recommended_speed_scale: float
    candidate_scores: dict
    stop_requested: bool
    reason: str
```

### Controller adapter

The adapter converts the selected lookahead point to:

```text
local_goal_error_rad = atan2(y_left, x_forward)
```

It creates existing `CandidateDirection` and `PerceptionResult` values.
`HybridReactiveController` remains responsible for expected `linear/angular`;
`CommandFilter` remains responsible for clamp, smoothing, and slew limiting.

## 6. Implementation Phases

### Phase 1 - SAM-TP output adapter

Add a production-facing adapter without rewriting `SamTpPredictor`.

Files:

- `src/earth_rover/core/types.py`
- `src/earth_rover/perception/sam_tp_adapter.py`
- `tests/test_sam_tp_adapter.py`

Gate:

- source-frame geometry is preserved;
- score semantics are not inverted or thresholded;
- invalid/timeout output has a stable failure reason;
- no controller or SDK dependency is introduced.

### Phase 2 - Constant-curvature trajectory sampler

Generate deterministic arcs and rover-footprint boundaries.

Files:

- `src/earth_rover/core/types.py`
- `src/earth_rover/planning/trajectory_sampler.py`
- `tests/test_trajectory_sampler.py`

For arc length `s` and curvature `k`:

```text
k = 0:
  x = s
  y = 0
  heading = 0

k != 0:
  heading = k*s
  x = sin(k*s)/k
  y = (1-cos(k*s))/k
```

Gate:

- at least seven configured candidates;
- exact straight path;
- positive-left and negative-right symmetry;
- explicit, validated rover width and safety margin;
- deterministic and read-only output.

### Phase 3 - Camera projection and calibration overlay

Add `planning/trajectory_projector.py` only after calibration fields and frame
conventions are measured. Project adjacent boundary cross-sections into filled
image-space corridor polygons.

Requirements:

- versioned intrinsics and distortion;
- versioned `T_camera_base`;
- calibrated source resolution;
- same-aspect intrinsics scaling only;
- rejection of invalid depth, horizon, and crop/aspect mismatch;
- debug overlay showing all boundaries and selected trajectory.

Gate:

- known calibration targets project within an agreed pixel/physical error;
- left/right projection and controller signs are manually confirmed;
- `camera.validated=false` prevents command-capable operation.

### Phase 4 - Traversability-only image-space scorer

Add `planning/trajectory_scorer.py`. Divide each corridor by rover-forward
distance into near/mid/far regions and sample the original SAM-TP score map.

Initial components:

```text
+ weighted mean traversability
+ lower quantile traversability
- near-field risk
- unknown/invalid ratio
- curvature cost
- previous-curvature change cost
```

A single-pixel minimum is not used. The fixed-width footprint corridor already
represents clearance; connected safe width remains diagnostic until replay
shows that an additional hard gate is necessary.

Gate:

- all-invalid and near-hard-risk cases stop;
- safe alternatives override a risky center path;
- deterministic tie-breaking and hysteresis work.

### Phase 5 - GPS-aware local planner

Add `planning/sam_tp_local_planner.py`. Reuse `bearing_deg`,
`normalize_angle_deg/rad`, `HeadingFilter`, and `WaypointManager`.

Development order:

1. traversability-only candidate selection;
2. GPS alignment added only among safety-approved candidates.

Default live/shadow policy for invalid GPS is STOP. Explicit offline
traversability-only degraded mode may be enabled and must be recorded as such.

Gate:

- GPS preference never overrides a hard safety rejection;
- expected acceptance scenarios for front/left/right goals and blocked paths
  pass;
- path selection does not oscillate on small score changes.

### Phase 6 - Existing controller and safety integration

Add `control/planner_controller_adapter.py`. Reuse
`HybridReactiveController`, `CommandFilter`, `EmergencyStopMonitor`,
`LatencyCompensator`, and existing recovery interfaces.

Priority remains:

```text
Emergency Stop
> Recovery
> Traversability Safety
> Goal Tracking
> Normal Driving
```

GPS heading and local-target steering contributions must be logged separately
because using both can over-steer. Recovery remains disabled in the first
planner replay.

Gate:

- bounded, finite, smooth expected commands;
- stale/invalid perception, projection, or sensors stop;
- no SDK control import is needed for offline tests.

### Phase 7 - Offline replay and visualization

Extend existing replay contracts and `LogOnlyControlSink`; do not create a
third independent replay architecture.

Input modes:

- HLS replay with explicitly fixed/synthetic heading input and
  `gps_valid=false`;
- `RunLogger.timeline.csv` replay with recorded GPS/heading/waypoint.

Reuse deterministic HLS selection, `ExistingHlsDecoder`, H.264 writer,
QuickTime-compatible settings, timestamp handling, and overwrite protection.

The video displays:

- original frame and SAM-TP score;
- every projected candidate corridor;
- selected path and candidate score components;
- GPS direction and goal input mode;
- expected command, latency, frame age, planner mode, and stop reason.

Gate:

- all rows contain `command_transmitted=false`;
- review video is H.264/yuv420p/10 FPS;
- human review approves calibration and path choices.

### Phase 8 - Delayed replay

Reuse the existing delayed-packet behavior and `urban_latency_2s.yaml`.
Compare `0`, `0.5`, `1`, and `2` seconds where practical; `0` versus `2`
seconds is mandatory.

Metrics:

- trajectory selection-change frequency;
- angular sign reversal;
- angular total variation;
- stop and stale-frame counts;
- path confidence;
- qualitative relation to recorded human command direction.

Delayed profiles reduce base speed and curvature range, strengthen hysteresis,
and stop after the configured stale limit.

### Phase 9 - SDK read-only shadow mode

Extend `ReadOnlySdkSource` and the existing SAM-TP dashboard. The initial shadow
launcher must expose no control, mission, or checkpoint-write path. It computes
planner/controller/safety output only for logs and visualization.

Validate:

- real SDK resolution and timestamp behavior;
- GPU throughput and end-to-end latency;
- GPS/camera left-right convention;
- angular command sign;
- stationary planner stability;
- stale frame and telemetry stop behavior.

Every record contains `command_transmitted=false`.

## 7. Configuration Plan

The first complete planner profile will be
`configs/urban_sam_tp_planner.yaml`. Machine paths and checkpoint paths remain
CLI/environment overrides.

Provisional offline values:

```yaml
sam_tp_planner:
  curvatures_per_m: [-0.80, -0.50, -0.25, 0.0, 0.25, 0.50, 0.80]
  horizon_m: 2.0
  sample_interval_m: 0.05
  rover_width_m: null
  safety_margin_m: null

camera:
  calibrated_resolution: null
  intrinsics: null
  distortion: null
  T_camera_base: null
  validated: false

trajectory_scoring:
  range_boundaries_m: [0.0, 0.6, 1.2, 2.0]
  range_weights: [3.0, 2.0, 1.0]
  lower_quantile: 0.20
  mean_weight: 0.35
  lower_quantile_weight: 0.35
  goal_alignment_weight: 0.20
  near_risk_weight: 0.50
  unknown_weight: 0.40
  curvature_weight: 0.05
  change_weight: 0.10
  safe_score_threshold: 0.50
  hard_stop_score_threshold: 0.25
  minimum_valid_ratio: 0.70
  hysteresis_margin: 0.05

speed_policy:
  minimum_scale: 0.0
  maximum_scale: 1.0
  high_curvature_scale: 0.5
  low_confidence_scale: 0.4

runtime:
  frame_stale_sec: 1.0
  shadow_mode: true
  send_commands: false
```

These values are test/replay starting points, not live-ready calibration.
`rover_width_m`, `safety_margin_m`, and camera geometry deliberately have no
default until measured.

## 8. Test Plan

### Unit tests

- valid SAM-TP conversion, dtype/range/shape, valid mask, timestamp, latency;
- NaN/Inf, timeout, invalid shape/range, and deterministic conversion;
- straight and zero-curvature paths;
- positive-left/negative-right symmetry;
- exact horizon endpoint and bounded sample spacing;
- invalid/duplicate curvature;
- invalid horizon, interval, rover width, and safety margin;
- in/out-of-image and behind-camera projection;
- invalid/mismatched calibration;
- unknown-only and low near-field corridors;
- all candidates invalid;
- GPS alignment and safety override;
- hysteresis and deterministic tie-breaking.

### Integration tests

- SAM-TP output through adapter, sampler, projector, scorer, and planner;
- planner output through existing controller and command filter;
- emergency stop override;
- deterministic `0s` and `2s` replay;
- H.264/yuv420p/10 FPS debug video;
- fake read-only SDK source with no write endpoint;
- `command_transmitted=false` schema enforcement.

GPU/checkpoint tests remain separate from checkpoint-free unit tests.

## 9. Acceptance Criteria

1. SAM-TP score maps are standardized without changing score direction.
2. At least seven deterministic constant-curvature paths are produced.
3. Every path carries rover-footprint corridor geometry.
4. Traversability-only selection works before GPS is introduced.
5. GPS affects only safety-approved candidates.
6. All-risky, all-invalid, stale, or failed perception results stop.
7. Hysteresis prevents unnecessary left/right oscillation.
8. The existing controller and filter generate bounded expected commands.
9. Existing safety always overrides planning and control.
10. Identical input manifests support comparable `0s` and `2s` replay.
11. Debug video exposes score, paths, command, timestamps, and stop reason.
12. SDK shadow mode has no command transmission path.

## 10. First Implementation Unit

The first bounded implementation includes only:

```text
SAM-TP output adapter
+ deterministic constant-curvature trajectory sampler
+ unit tests
```

Files:

- `src/earth_rover/core/types.py`
- `src/earth_rover/perception/sam_tp_adapter.py`
- `src/earth_rover/planning/trajectory_sampler.py`
- `tests/test_sam_tp_adapter.py`
- `tests/test_trajectory_sampler.py`

Verification:

```bash
pytest tests/test_sam_tp_adapter.py tests/test_trajectory_sampler.py
pytest tests/test_sam_tp_reproduction.py tests/test_sam_tp_sdk_shadow.py
```

This unit contains no camera projection, image scoring, GPS selection,
controller integration, replay modification, SDK extension, BEV, or rover
motion.

## 11. Calibration and Live Deployment Blockers

The repository does not currently contain verified values for:

- camera intrinsics and distortion;
- camera height, pitch, yaw, roll, and translation relative to base;
- SDK frame crop/resolution policy;
- rover width and safety clearance;
- practical minimum turning radius;
- curvature-to-`linear/angular` response;
- acceptable projection error;
- required end-to-end planning rate.

FrodoBots ride data reportedly contains GPS and IMU files, but the current HLS
manifest/review pipeline reads only front timestamps and human control. The
exact Dell GPS/IMU schema and timestamp alignment must be audited before
claiming recorded GPS replay.

SAM-TP has no obstacle class or depth and cannot certify clearance on curbs,
overhangs, or non-planar terrain. Image-space projection still relies on
calibrated camera/base geometry. IMU roll/pitch must not be applied until its
frame convention is confirmed; otherwise camera extrinsics may be corrected
twice.

The frozen upstream commit has no root license file. Offline research use does
not resolve competition or deployment permission. Licensing, calibration,
projection overlay review, command sign verification, delayed replay, SDK
shadow, and emergency-stop tests are mandatory blockers before any separately
authorized live-motion implementation.
