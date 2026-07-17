# Offline Training Workflow

## FrodoBots-2K Phase 1 Manifest

Build a bounded three-ride manifest without decoding or extracting video frames:

```bash
.venv/bin/python training/build_frodobots_2k_manifest.py \
  --dataset-root ../datasets_2k/output_rides_0 \
  --output-dir ../datasets_2k/manifests/frodobots_2k_phase1/dry_run_3rides \
  --max-rides 3 \
  --control-tolerance-ms 100
```

The command writes `manifest.csv` and `alignment_report.json`. The output must remain outside the immutable `output_rides_0` directory and outside Git. Complete this validation before implementing HLS frame decoding or model training.

On the Dell Ubuntu host, run the complete focused verification with no arguments:

```bash
./scripts/verify_phase1_manifest.sh
```

The script uses `/home/asl/datasets/output_rides_0`, writes generated files outside Git under `/home/asl/datasets/manifests/`, and verifies that the raw dataset metadata is unchanged.

## FrodoBots-2K Phase 2 HLS Verification

On the Dell Ubuntu host, run the lazy HLS loader and visual verification with no arguments:

```bash
./scripts/verify_phase2_hls_loader.sh
```

The script runs focused loader tests, decodes 20 deterministic manifest samples, verifies a `4x3x224x224` DataLoader batch, checks repeat access, and writes `aligned_samples.jpg` plus `hls_verification_report.json` under `/home/asl/datasets/outputs/frodobots_2k_phase2/`. Inspect the contact sheet manually before treating image-label alignment as validated.

Before Phase 3, run the full semantic and edge-case audit on Dell:

```bash
./scripts/audit_phase2_alignment.sh
```

The audit builds a read-only manifest for all rides, creates early/middle/late temporal strips from at least five rides, prioritizes HLS-discontinuity rides, visualizes LEFT/RIGHT and available REVERSE samples, and reports transform, batch, monotonicity, and decode-failure details. Its automatic result remains `CONDITIONAL PASS` until `left_strips.jpg` and `right_strips.jpg` are reviewed by a person.

## Phase 3 ResNet18 Tiny Overfit

After Phase 2 passes, run the 200-sample GPU overfit gate on Dell:

```bash
./scripts/verify_phase3_tiny_overfit.sh
```

The script selects 40 samples from each action class, caches only those decoded tensors in memory, fine-tunes an ImageNet-pretrained ResNet18, and verifies loss reduction, at least 95% training accuracy, checkpoint reload, deterministic inference, and raw-dataset immutability. The first run may download the torchvision ResNet18 weights. Outputs remain outside Git under `/home/asl/datasets/outputs/frodobots_2k_phase3/tiny_overfit/`.

After the tiny-overfit gate passes, run the bounded 10/2/2 ride-level baseline:

```bash
./scripts/run_phase3_small_baseline.sh
```

The run uses at most 250 samples per ride, selects the best checkpoint by validation macro F1, evaluates the held-out test rides once, and writes `held_out_test_predictions.mp4` with ground truth, prediction, confidence, and control overlays. Metrics, the exact ride split, class distributions, and confusion matrices are stored in `small_baseline_report.json` under `/home/asl/datasets/outputs/frodobots_2k_phase3/small_baseline/`.

## Traversability Pseudo-Label Pilot

The action baseline remains unchanged. Traversability pseudo-labeling is a separate research-only workflow using `nvidia/segformer-b0-finetuned-ade-512-512` as an annotation draft, never as verified ground truth or a rover controller.

On Dell, install the optional pinned dependency and preserve its resolver report:

```bash
./scripts/setup_traversability_pilot.sh
```

Then build the 40-frame, eight-ride default review bundle:

```bash
./scripts/run_traversability_pilot.sh
```

Paths can be overridden with `DATASET_ROOT`, `MANIFEST_PATH`, `BUNDLE_ROOT`, `SAMPLE_COUNT`, `MAX_RIDES`, `MINIMUM_SEPARATION_SECONDS`, and `SEED`. The default bundle is `$HOME/datasets/review_bundles/traversability_pilot_v1/`. Open `gallery.html` on the Mac and edit reviewer columns in `review.csv`. Do not train until the reviewed CSV and any corrected masks pass `training/validate_traversability_review.py` and the user explicitly approves them.

Copy the self-contained bundle by running this on the Mac, replacing both placeholders:

```bash
rsync -ah --progress \
  asl@<DELL_TAILSCALE_IP>:/home/asl/datasets/review_bundles/traversability_pilot_v1/ \
  <MAC_DESTINATION>/traversability_pilot_v1/
```

## Traversability Dataset v1 Annotation Pilot

After reviewing the pseudo-label pilot, prepare exactly 20 images for manual four-class annotation without rerunning SegFormer:

```bash
./scripts/run_traversability_annotation_pilot.sh
```

The default output is `$HOME/datasets/generated/traversability_dataset_v1/pilot_20/`. Follow `docs/training/traversability_dataset_v1_annotation.md` for the CVAT workflow and Dell-only import/validation commands. The required IDs are `0 IGNORE`, `1 ON_ROAD`, `2 OFF_ROAD`, and `3 OBSTACLE`. Do not expand the dataset or train from these masks until the user reviews and approves the completed pilot.

After explicit approval of the imported 20-image pilot, build the additional 100-image CVAT bundle on Dell:

```bash
./scripts/build_traversability_annotation_100.sh
```

This performs bounded pseudo inference on 240 manifest samples for selection and seed drafts, not full-dataset inference. It excludes the approved pilot by provenance, time, and visual hash; caps each ride at five selected images; and writes the new bundle outside Git under `$HOME/datasets/generated/traversability_dataset_v1/annotation_100_v1/`. Stop after generation and annotate all 100 images manually before any training.

## Berkeley-FrodoBots-7K Probe

Do not download the full Berkeley-FrodoBots-7K dataset during initial work. It is too large for local iteration.

## First Probe

Install optional training dependencies:

```bash
.venv/bin/pip install datasets huggingface_hub
```

Authenticate once for the gated dataset:

```bash
huggingface-cli login
```

Stream a small sample:

```bash
.venv/bin/python training/explore_berkeley_frodobots_7k.py --max-rows 200
```

Outputs:

- `datasets/berkeley_7k_probe/summary.json`
- `datasets/berkeley_7k_probe/sample_rows.jsonl`
- `datasets/berkeley_7k_probe/parsed_actions.csv`

## Decision Gate

After the probe, inspect:

- which action key is present: `action_mbra`, `action`, or `action_original`
- whether actions are exposed as numeric arrays or encoded payloads
- whether `__url__` groups samples by shard or source file
- whether image paths are available directly or require video extraction

Only after this should we build the real PyTorch `Dataset`.
