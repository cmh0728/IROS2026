# Traversability Dataset v1 Annotation

This pilot defines human-reviewed semantic masks independently from the existing ResNet18 action baseline and the three-class pseudo-label experiment. It does not authorize training or rover integration.

## Label Contract

Masks are single-channel PNG files at the original image dimensions. The only valid pixel values are:

| ID | Class | Use |
|---:|---|---|
| 0 | `IGNORE` | Uncertain or excluded pixels; ignored by a future loss. |
| 1 | `ON_ROAD` | Traversable paved ground. |
| 2 | `OFF_ROAD` | Traversable unpaved ground. |
| 3 | `OBSTACLE` | No-entry regions such as people, vehicles, furniture, poles, walls, curbs, and stairs. |

Mark ambiguous ground, severe shadow or reflection, sky, and rover hood as `IGNORE`. Label terrain by physical traversability, not route preference. A future planner may assign a higher cost to `OFF_ROAD`; the segmentation mask must not encode that preference.

Use `ON_ROAD` only for visibly usable pavement. Use `OFF_ROAD` for visibly usable grass, dirt, or gravel, but mark uncertain terrain and unclear boundaries `IGNORE`. People, vehicles, benches, poles, walls, curbs, and stairs are `OBSTACLE`. Treat narrow paths, backlight, lens distortion, and partly occluded boundaries conservatively.

## Dell: Build the Pilot

The completed 40-frame pseudo-label bundle is a read-only candidate pool. Build exactly 20 annotation images without running SegFormer again:

```bash
./scripts/run_traversability_annotation_pilot.sh
```

The default output is `$HOME/datasets/generated/traversability_dataset_v1/pilot_20/`. Override `SOURCE_BUNDLE`, `OUTPUT_DIR`, `MINIMUM_SEPARATION_SECONDS`, or `SEED` only when needed. Generated images and masks remain outside Git.

Copy the portable pilot from Dell to Mac using the macOS-compatible progress option:

```bash
rsync -ah --progress \
  asl@<DELL_TAILSCALE_IP>:/home/asl/datasets/generated/traversability_dataset_v1/pilot_20/ \
  "$HOME/Desktop/traversability_dataset_v1_pilot_20/"
```

## Mac: Annotate in CVAT

CVAT is selected because its standard [`Segmentation Mask 1.1`](https://docs.cvat.ai/docs/manual/advanced/formats/format-smask/) format directly imports and exports pixel masks. Label Studio would require an additional JSON/RLE conversion path. The generated seed archive contains the standard `labelmap.txt`, `ImageSets/Segmentation/default.txt`, `SegmentationClass/`, and `SegmentationObject/` entries.

1. Create a CVAT task and upload all files from `images/`.
2. Define `IGNORE`, `ON_ROAD`, `OFF_ROAD`, and `OBSTACLE` in the exact order and colors shown in `cvat_labelmap.txt`.
3. Annotate from scratch, or import `cvat_seed_annotations.zip` as `Segmentation Mask 1.1`.
4. Treat every seed pixel as unverified. The seed never distinguishes `OFF_ROAD` and must be corrected manually.
5. Export `Segmentation Mask 1.1` without source images and copy the ZIP back to Dell.

```bash
scp <MAC_CVAT_EXPORT_ZIP> \
  asl@<DELL_TAILSCALE_IP>:/home/asl/datasets/generated/traversability_dataset_v1/cvat_export.zip
```

The Mac is for labeling only. Do not run dataset processing, validation, inference, or training there.

## Dell: Import and Validate

For the reviewed export `traversability_pilot_20_reviewed.zip`, run the complete focused import and verification workflow:

```bash
./scripts/import_validate_traversability_review.sh
```

The script reads only `SegmentationClass/`. It parses `labelmap.txt` by label name, maps `background` to `IGNORE=0`, rejects unknown labels and contract-color conflicts, and writes normalized results under `pilot_20/reviewed_import/`. It preserves the original ZIP and checks the raw FrodoBots metadata fingerprint.

To run the two CLIs separately after inspecting `--help`:

```bash
python3 training/import_cvat_traversability_masks.py \
  --bundle "$HOME/datasets/generated/traversability_dataset_v1/pilot_20" \
  --cvat-export "$HOME/datasets/generated/traversability_dataset_v1/pilot_20/traversability_pilot_20_reviewed.zip" \
  --output-dir "$HOME/datasets/generated/traversability_dataset_v1/pilot_20/reviewed_import" \
  --expected-count 20

python3 training/validate_traversability_dataset_v1.py \
  --bundle "$HOME/datasets/generated/traversability_dataset_v1/pilot_20" \
  --masks-dir "$HOME/datasets/generated/traversability_dataset_v1/pilot_20/reviewed_import/masks" \
  --report-path "$HOME/datasets/generated/traversability_dataset_v1/pilot_20/reviewed_import/validation_report.json"
```

The validator rejects missing or extra masks, duplicate sample IDs, filename mismatches, dimension mismatches, multi-channel final masks, and IDs outside `0..3`. It reports per-image and overall class distributions plus all-IGNORE and single-class warnings. Stop after validation and review `overlay_contact_sheet.jpg` or `review.html` before expanding the dataset or training a model.

Reviewed outputs are written under `pilot_20/reviewed_import/`: normalized masks in `masks/`, colored mask previews in `mask_visualizations/`, overlays in `overlays/`, per-image statistics in `per_image_statistics.csv`, and the full validator result in `validation_report.json`. Copy `overlay_contact_sheet.jpg` to the Mac for the final human gate; do not train after an automated PASS alone.

## Approved 100-Image Expansion

After explicit approval of the 20-image pilot, prepare the additional annotation bundle on Dell:

```bash
./scripts/build_traversability_annotation_100.sh
```

The workflow reuses the full manifest, lazy HLS decoder, conservative SegFormer pseudo-label pipeline, and the four-class bundle writer. It runs pseudo inference on a bounded 240-frame candidate pool only, then selects exactly 100 images. It does not run inference over the full dataset and does not train a model.

Selection excludes exact provenance matches, nearby timestamps, and visual near-duplicates of the approved 20 images. A 64-bit difference hash suppresses near-identical candidates, each ride is capped at five images, and selected timestamps from the same ride are separated by at least 10 seconds. Sky-dominant, severely blurred, low-information, and high-unknown-without-ground candidates are reported separately and do not count toward the 100 images.

The default bundle is `$HOME/datasets/generated/traversability_dataset_v1/annotation_100_v1/`. It uses `trav_v1_add_#####` sample IDs so the approved `trav_v1_#####` pilot remains unchanged. `cvat_seed_annotations.zip` remains an unverified draft: old traversable maps to `ON_ROAD`, old non-traversable maps to `OBSTACLE`, old unknown maps to `IGNORE`, and `OFF_ROAD` is never auto-generated. Stop after bundle generation and complete all 100 annotations manually in CVAT.
