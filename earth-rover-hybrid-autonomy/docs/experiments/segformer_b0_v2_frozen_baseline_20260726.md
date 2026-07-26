# SegFormer-B0 v2 Frozen Baseline

Freeze date: 2026-07-26

This document logically freezes the existing SegFormer-B0 v2 traversability
baseline. It does not move, copy, modify, or re-run the dataset, checkpoint, or
experiment outputs.

## Model contract

- Architecture: SegFormer-B0 initialized from the approved v1 three-class
  checkpoint; the original ADE20K head is not used at inference.
- Input: aspect-ratio-preserving letterbox to 512, ImageNet normalization.
- Stored annotation IDs: `0 IGNORE`, `1 ON_ROAD`, `2 OFF_ROAD`, `3 OBSTACLE`.
- Training IDs: `255 IGNORE`, `0 ON_ROAD`, `1 OFF_ROAD`, `2 OBSTACLE`.
- Model labels: 3; loss ignore index: 255.
- Configuration: `configs/traversability_segformer_b0_v2.yaml`.

## Frozen dataset and split

- Dataset version: `traversability_dataset_v2_approved_153`.
- Expected sample count: 153 (approved v1 120 + manually reviewed v2 33).
- Dataset root on Dell:
  `~/datasets/generated/traversability_dataset_v2/approved_153_v2`.
- Manifest: `approved_153_v2/manifest.csv`.
- Merge report: `approved_153_v2/merge_report.json`.
- Split report: `approved_153_v2/split_report.json`.
- Fixed v1 train/validation/test manifests:
  `approved_153_v2/fixed_v1_splits/{train,validation,test}.csv`.
- The v1 validation and test assignments remain fixed. New manual samples use
  ride-grouped `train` and `new_holdout` assignments.

The 153 count and split rules are enforced by tracked builder/tests. The
generated Dell manifests are not present in this Git checkout, so their hashes
and actual per-split counts are unverified here and must be read from the Dell
reports above.

## Frozen experiment

- Best checkpoint:
  `~/datasets/experiments/traversability_segformer_b0_v2/full_training/segformer_b0_best.pt`.
- Training report:
  `~/datasets/experiments/traversability_segformer_b0_v2/full_training/experiment_report.json`.
- v1/v2 comparison:
  `~/datasets/experiments/traversability_segformer_b0_v2/v1_v2_comparison/comparison_report.json`.
- Evaluation overlays:
  `~/datasets/experiments/traversability_segformer_b0_v2/v1_v2_comparison/reviews/`.
- Three-panel video review:
  `~/datasets/review_bundles/traversability_video_review_v2/`.
- Video review manifest:
  `~/datasets/review_bundles/traversability_video_review_v2/review_manifest.json`.

The generated reports and checkpoint are intentionally outside Git and were not
available on this Mac during the freeze commit. Consequently, best epoch,
checkpoint SHA-256, test mIoU, new-holdout mIoU, regression status, latency, and
video statistics are recorded as **unverified in this checkout**. Read those
values directly from the files above; do not copy values from conversation
history.

## Provenance and role

- Dataset v2 workflow commit: `972084e` (`Add traversability dataset v2 workflow`).
- Checkpoint validation fix: `890b7ce` (`Fix SegFormer checkpoint validation`).
- Video review commit: `5b3bce8` (`Add traversability v2 video review`).
- Freeze revision: the commit containing this document.
- Reason: preserve a reproducible lightweight reference while evaluating
  alternative perception models.
- Future role: baseline, fallback, challenger comparison, and possible student
  model. It is not deleted or replaced by SAM-TP.

SegFormer and SAM-TP do not share the same output contract: this baseline is a
three-class semantic traversability model, while official SAM-TP predicts a
continuous binary traversable-region score. Their metrics must not be compared
as if they measured the same label space.
