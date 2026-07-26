#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$WORKSPACE_ROOT/external/GENIE-SAMTP}"
ENV_NAME="${ENV_NAME:-sam_tp_repro}"
UPSTREAM_URL="https://github.com/jiaming-ai/GENIE-SAMTP.git"
UPSTREAM_BRANCH="master"
UPSTREAM_COMMIT="728aee296cf44288356de683b1948f18b05917d6"
CHECKPOINT_RELATIVE="sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt"
CHECKPOINT_URL="https://drive.google.com/drive/folders/190yHH-TcfQVoByZeB1809sPIR62CsBD1?dmr=1&ec=wgc-drive-hero-goto"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; this Dell reproduction requires an NVIDIA GPU." >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required because the official environment.yml is present." >&2
  echo "Install Miniconda/Conda, then rerun this script." >&2
  exit 2
fi

if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
  if [[ -e "$UPSTREAM_ROOT" ]]; then
    echo "ERROR: Non-Git path already exists: $UPSTREAM_ROOT" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$UPSTREAM_ROOT")"
  git clone --branch "$UPSTREAM_BRANCH" "$UPSTREAM_URL" "$UPSTREAM_ROOT"
fi

actual_remote="$(git -C "$UPSTREAM_ROOT" remote get-url origin)"
if [[ "$actual_remote" != "$UPSTREAM_URL" ]]; then
  echo "ERROR: Unexpected upstream remote: $actual_remote" >&2
  exit 2
fi
if [[ -n "$(git -C "$UPSTREAM_ROOT" status --porcelain)" ]]; then
  echo "ERROR: Upstream checkout is dirty; preserve or remove its changes manually." >&2
  exit 2
fi
git -C "$UPSTREAM_ROOT" fetch origin "$UPSTREAM_BRANCH"
git -C "$UPSTREAM_ROOT" checkout --detach "$UPSTREAM_COMMIT"
if [[ "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]]; then
  echo "ERROR: Could not pin the official upstream commit." >&2
  exit 2
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env create --name "$ENV_NAME" --file "$UPSTREAM_ROOT/environment.yml"
fi
conda run --no-capture-output -n "$ENV_NAME" python -c \
  "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

if ! conda run --no-capture-output -n "$ENV_NAME" python -c "import torch, torchvision" >/dev/null 2>&1; then
  if [[ "${INSTALL_TORCH:-false}" != "true" ]]; then
    cat >&2 <<EOF
ERROR: PyTorch is absent from the independent '$ENV_NAME' environment.
The official environment.yml intentionally omits PyTorch.

Verify the Dell driver first with: nvidia-smi
Then rerun explicitly, without changing the existing Earth Rover environment:

INSTALL_TORCH=true TORCH_VERSION=2.7.1 TORCHVISION_VERSION=0.22.1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
  ./scripts/setup_sam_tp_reproduction.sh
EOF
    exit 3
  fi
  TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
  TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX_URL"
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e "$UPSTREAM_ROOT"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "pytest==8.4.2"

checkpoint="$UPSTREAM_ROOT/$CHECKPOINT_RELATIVE"
if [[ ! -f "$checkpoint" ]]; then
  cat >&2 <<EOF
SAM-TP environment and upstream checkout are ready.
The official checkpoint requires manual Google Drive download:

URL: $CHECKPOINT_URL
Expected filename: checkpoint_2.pt
Place it at:
$checkpoint

After placement, verify it with:
sha256sum "$checkpoint"

Then rerun this script and continue with scripts/run_sam_tp_reproduction.sh.
EOF
  exit 4
fi

echo "SAM-TP setup ready"
echo "Environment: $ENV_NAME"
echo "Upstream: $UPSTREAM_ROOT"
echo "Commit: $(git -C "$UPSTREAM_ROOT" rev-parse HEAD)"
echo "Checkpoint: $checkpoint"
