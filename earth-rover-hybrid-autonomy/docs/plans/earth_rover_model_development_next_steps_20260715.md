# Earth Rover Model Development — Next Steps

Last updated: 2026-07-15

## 1. Decision Summary

Development will begin with a lightweight ResNet18 baseline. This is not the final rover navigation model. Its purpose is to verify the complete dataset, training, evaluation, inference, and delayed-replay pipeline before adopting a more complex goal-conditioned navigation model.

The agreed progression is:

```text
ResNet18 image-action baseline
→ temporal goal-conditioned baseline
→ ViNT/GNM/LogoNav-inspired local waypoint policy
→ latency-aware controller and safety integration
→ low-speed live-rover validation
```

The production system remains a hybrid autonomy stack. A learned model proposes local navigation behavior, while GPS waypoint logic, delay compensation, control, safety, and recovery remain explicit modules.

## 2. Current Data Checkpoint

The inspected FrodoBots-2K sample is an already extracted ride collection:

- Path name: `datasets_2k/output_rides_0`
- Size: approximately 21 GB
- Rides: 231
- Files: 20,263
- Empty files: none found
- Images are stored as HLS video, not individual image files.
- Sensor and command data are stored in CSV files.

Available modalities include:

- front H.264 MPEG-TS video at approximately 20 Hz;
- rear H.264 MPEG-TS video at approximately 20 Hz;
- `linear`, `angular`, four wheel RPM values, and timestamp at approximately 10 Hz;
- GPS latitude/longitude at approximately 1 Hz;
- IMU JSON samples with mixed internal rates;
- speaker audio for all rides and microphone audio for a subset.

Timestamp units are mixed:

- camera, control, and audio: Unix seconds;
- GPS and outer IMU timestamp: Unix milliseconds;
- inner IMU samples: Unix seconds.

Important quality findings:

- front/control alignment within 100 ms: approximately 58.05%;
- front/control alignment within 200 ms: approximately 60.24%;
- median overlap of all major modalities per ride: approximately 31.4%;
- duplicate and reversed control timestamps exist;
- camera gaps and HLS discontinuities exist;
- GPS jumps, RPM outliers, and IMU gaps exist;
- action labels are heavily imbalanced;
- complete decode integrity of every TS segment has not been verified.

The initial model must therefore use filtered, timestamp-aligned samples rather than every camera frame.

## 3. First Baseline Model

### Purpose

Verify that the complete machine-learning pipeline works correctly:

```text
HLS video decoding
→ timestamp alignment
→ action-label generation
→ PyTorch Dataset/DataLoader
→ RTX 4070 training
→ checkpoint save/load
→ validation and metrics
→ held-out inference
→ prediction visualization
→ delayed replay
```

### Architecture

- Backbone: ImageNet-pretrained ResNet18
- Input: one front RGB frame resized to 224×224
- Output classes:
  - `STOP`
  - `FORWARD`
  - `LEFT`
  - `RIGHT`
  - `REVERSE`
- Loss: weighted cross entropy
- Split: strictly by ride, never by randomly mixed frames

Existing action-label logic should be reused if the repository already contains it. Thresholds must not be duplicated under a second implementation.

### Initial exclusions

The first baseline will not include:

- raw GPS input;
- IMU input;
- rear-camera input;
- temporal Transformer, GRU, or LSTM;
- diffusion policy;
- neural world model;
- explicit traversability segmentation;
- direct live-rover control.

This baseline is a pipeline and representation check, not the production controller.

## 4. Phase 1 — Manifest and Data Validation

### Goal

Create a deterministic, immutable manifest connecting valid front-camera frames to trustworthy control labels.

### Required manifest fields

- `ride_id`
- front playlist or recording reference
- front frame ID
- normalized front timestamp in Unix seconds
- matched control timestamp in Unix seconds
- control delta in milliseconds
- `linear`
- `angular`
- action class

### Rules

- Never modify the raw dataset.
- Normalize every timestamp to Unix seconds.
- Match control data only within the same ride.
- Use nearest-control matching.
- Start with a 100 ms control tolerance.
- Do not silently resolve conflicting duplicate control timestamps.
- Split sequences at timestamp reversal and HLS discontinuity boundaries where necessary.
- Keep `ride_id` so later train/validation/test splitting is leakage-free.
- Support a dry run on two or three rides before processing all rides.
- Do not pre-decode every video or extract every frame.

### Tests

- seconds/milliseconds timestamp normalization;
- nearest-control matching;
- 100 ms tolerance rejection;
- duplicate control timestamp handling;
- reversed timestamp handling;
- action-label mapping;
- prevention of cross-ride matching.

### Completion criteria

- focused tests pass;
- manifest dry run succeeds on two or three rides;
- report includes valid/invalid sample counts and exclusion reasons;
- report includes class distribution and control-delta p50/p95/max;
- generated dataset artifacts remain outside Git.

## 5. Phase 2 — HLS Loader and Visual Verification

### Goal

Read only the requested HLS frame at training time and verify that every image-label pair is semantically plausible.

### Tasks

- implement lazy HLS frame decoding;
- create the PyTorch Dataset and DataLoader;
- return image tensor, action label, ride ID, and alignment metadata;
- visualize at least 20 aligned samples;
- overlay ride ID, frame timestamp, control delta, `linear`, `angular`, and action class;
- detect unreadable frames and report them rather than silently substituting data.

### Completion criteria

- batches have the expected tensor shape;
- repeated sample access is deterministic;
- visualization confirms that labels broadly match observed motion intent;
- unreadable or missing samples are rejected safely;
- no whole-dataset frame extraction is required.

## 6. Phase 3 — Tiny Overfit and Small Baseline

### Tiny-overfit test

- Use approximately 100–300 verified samples.
- Train until the model can strongly overfit this small subset.
- Verify decreasing loss, high training accuracy, checkpoint save/load, and reproducible inference.

Failure to overfit is treated as a pipeline bug until proven otherwise.

### Small ride-level experiment

Start with a limited experiment such as:

```text
Train: 10 rides
Validation: 2 separate rides
Test: 2 separate rides
```

Metrics:

- macro F1;
- balanced accuracy;
- per-class precision and recall;
- confusion matrix;
- overall accuracy as a secondary metric.

Class imbalance must be handled, initially with weighted cross entropy. `FORWARD` and `STOP` predictions alone must not be interpreted as useful navigation performance.

## 7. Phase 4 — Held-Out Inference and Delayed Replay

### Offline inference

Save:

- predicted class and confidence;
- ground-truth class;
- original image with prediction overlay;
- incorrect and low-confidence examples;
- per-ride metric summary.

### Delayed replay

Use the existing two-second latency profile and delayed-replay tooling. Verify:

- actual model inference latency;
- stale-image detection;
- no indefinite command hold;
- confidence-based stop behavior;
- safe handling of missing frames;
- no live rover command is sent.

Open-loop classification accuracy is not evidence that the rover can navigate safely in closed loop.

## 8. Phase 5 — Goal-Conditioned Temporal Baseline

After the ResNet18 pipeline is verified, extend it to:

```text
recent front images (3–5)
+ relative waypoint direction
+ previous control
+ optional speed/RPM
→ short future action sequence
```

Do not feed raw latitude and longitude directly to the model. The waypoint manager should compute:

- `sin(heading_error)`;
- `cos(heading_error)`;
- normalized distance to goal.

GPS and IMU remain primarily responsible for goal tracking, state estimation, delay compensation, and safety supervision.

## 9. Target Navigation Architecture

The longer-term learned policy should be inspired by GNM, ViNT, and LogoNav/MBRA:

```text
front image history
+ relative GPS waypoint direction
+ predicted current motion state
→ local waypoint sequence
+ stop probability
+ risk/confidence
```

A classical latency-aware controller converts local waypoints into SDK `linear` and `angular` commands. The model does not have final authority over live control.

NoMaD or another diffusion policy remains a later comparison candidate, not the initial implementation.

## 10. Delay Compensation and Safety

The rover receives delayed observations, so inference output must not be applied as if the observation described the current state.

Record at least:

- sensor capture time;
- SDK receive time;
- inference start/end time;
- command send time;
- next observation capture time.

Use command history, speed/RPM, GPS, and IMU heading to predict movement during observation age. Start with a kinematic predictor or EKF, not a neural visual world model.

The safety supervisor should monitor:

- sensor and image age;
- distance-to-goal progress;
- waypoint heading error;
- cross-track error when available;
- predicted-versus-observed motion discrepancy;
- RPM/command consistency;
- model confidence;
- communication loss.

Operational states should remain explicit:

```text
NORMAL → CAUTION → STOP → RECOVERY
```

Commands must be bounded, short, and latency-aware. If observations become stale or state disagreement becomes unsafe, the rover must stop rather than continue an old action.

## 11. Camera Roles

- Front camera: normal navigation and traversability inference.
- Rear camera: recovery and reverse-safety checks.
- Do not add the rear camera to every normal-navigation inference until evidence shows that it improves performance enough to justify additional synchronization and compute cost.

## 12. Compute and Storage Workflow

- MacBook: primary development, Git management, SSH control, inspection, and lightweight smoke tests.
- RTX 4070 Ubuntu laptop: training and inference.
- NAS: raw/processed datasets, manifests, checkpoints, experiments, and shared results.
- Tailscale + SSH: remote access from MacBook to the Ubuntu laptop.
- Git: source-code synchronization; source code must not live primarily on the NAS.
- Use NAS over the laboratory LAN when possible.
- Cache active shards locally on the RTX laptop if NAS reads become a GPU data-loading bottleneck.

## 13. Tomorrow's Starting Point

Start with Phase 1 only:

1. Resume the existing Codex session in the autonomy repository.
2. Inspect `AGENTS.md`, `git status`, current dataset utilities, tests, and action-label logic.
3. Save the completed FrodoBots-2K audit in the repository's appropriate documentation area.
4. Implement and test the immutable manifest builder.
5. Run a two-to-three-ride dry run.
6. Review the generated statistics and diff before beginning HLS decoding or model training.

Do not begin with ViNT, NoMaD, GPS/IMU fusion, a world model, or live-rover integration. Complete each phase and verify its success criteria before moving to the next.
