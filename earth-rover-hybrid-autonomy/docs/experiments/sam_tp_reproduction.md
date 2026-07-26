# Official SAM-TP Reproduction

This workflow reproduces only the SAM-TP perception model from the official
GeNIE release. It does not port the BEV planner, train SAM-TP, modify SegFormer,
or connect either model to the SDK or rover control.

## Frozen upstream

- Repository: `https://github.com/jiaming-ai/GENIE-SAMTP.git`
- Default branch: `master`
- Commit: `728aee296cf44288356de683b1948f18b05917d6`
- Inference config:
  `sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml`
- Matching training config:
  `sam2/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml`
- Expected checkpoint:
  `sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt`
- Manual checkpoint source:
  `https://drive.google.com/drive/folders/190yHH-TcfQVoByZeB1809sPIR62CsBD1?dmr=1&ec=wgc-drive-hero-goto`

The upstream README currently shows a different clone URL. Use the URL above.
The official repository has no root `LICENSE` file at the frozen commit, even
though inherited SAM2 source headers refer to one. Public source availability
is not a grant of unrestricted reuse; resolve licensing before redistribution
or deployment.

## Independent Dell setup

The setup script clones upstream under the ignored workspace `external/`
directory and creates a separate Conda environment. It does not upgrade the
Earth Rover/SegFormer environment.

```bash
cd ~/IROS2026/earth-rover-hybrid-autonomy
./scripts/setup_sam_tp_reproduction.sh
```

The official `environment.yml` omits PyTorch. If the script reports that
PyTorch is absent, inspect `nvidia-smi`, then explicitly install the known Dell
baseline runtime in the new environment:

```bash
INSTALL_TORCH=true \
TORCH_VERSION=2.7.1 \
TORCHVISION_VERSION=0.22.1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
./scripts/setup_sam_tp_reproduction.sh
```

This uses a stable CUDA 11.8 wheel and does not install the README's CUDA 12.8
nightly example. The NVIDIA driver's displayed CUDA compatibility version and
PyTorch's bundled CUDA runtime are recorded separately.

Download `checkpoint_2.pt` manually from the official Google Drive folder and
place it at the exact expected path printed by the setup script. Do not rename,
convert, or edit the checkpoint.

## Strict reproduction

Run one image benchmark plus one deterministic 30-second FrodoBots segment:

```bash
RUN_ID=sam_tp_official_$(date -u +%Y%m%dT%H%M%SZ) \
DATASETS=0 \
RIDES_PER_DATASET=1 \
SECONDS_PER_RIDE=30 \
./scripts/run_sam_tp_reproduction.sh
```

`PROJECT_PYTHON` defaults to Dell's `python3` and runs the existing Earth Rover
test suite without changing that environment. Override it only when the
project's installed Python executable has a different path.

The frozen config uses the same selector seed, five rides, 60-second windows,
10 FPS, edge margin, and gap limit as the existing SegFormer v2 review. To
reproduce that full comparable selection across all three dataset roots:

```bash
RUN_ID=sam_tp_all_$(date -u +%Y%m%dT%H%M%SZ) \
DATASETS="0 1 2" \
RIDES_PER_DATASET=5 \
SECONDS_PER_RIDE=60 \
./scripts/run_sam_tp_reproduction.sh
```

The generated `review_manifest.json` is the authoritative input-selection
record. The tracked repository does not contain the prior generated SegFormer
manifest, so equality must be confirmed on Dell by comparing each dataset's
`selected_segments` entries before using the videos as paired model evidence.

Outputs:

- Compatibility and benchmark:
  `~/datasets/experiments/sam_tp_reproduction/<RUN_ID>/`
- QuickTime H.264 review:
  `~/datasets/review_bundles/sam_tp_reproduction/<RUN_ID>/`
- Strict config/checkpoint report: `compatibility_report.json`
- Single-image logits and score: `single_image/{raw_logits,traversability_score}.npy`
- Single-image provenance/benchmark: `single_image/metadata.json`
- Aggregate result: `reproduction_report.json` and `reproduction_report.md`
- Video sampling/provenance: `review_manifest.json`
- Per-frame timings/scores: `<dataset>/frame_statistics.jsonl`

The strict gate compares shape-defining fields in the training and inference
configs, compares every model/checkpoint key and tensor shape, and performs a
strict state-dict load. Any missing, unexpected, or mismatched key stops before
inference. The official load path remains
`SAM_TP -> build_sam2 -> _load_checkpoint`; no `strict=False`, key rewrite, or
randomly initialized fallback is used.

## Output semantics

The official wrapper places three positive prompts along the image bottom and
requests unthresholded mask logits. This project stores those logits unchanged
and separately computes `sigmoid(raw_logits)` as a continuous score. Higher
scores mean stronger membership in that prompted traversable region. Video
colors are red for high scores and blue for low scores. The heatmap is
visualization-only and no threshold is hidden in the adapter.

SAM-TP's continuous binary output and SegFormer's semantic classes have
different meanings. Do not compare their mIoU directly or connect SAM-TP output
to the planner before a shared hard-set review establishes the intended label
semantics.
