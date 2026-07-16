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
