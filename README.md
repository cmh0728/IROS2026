# IROS 2026 Earth Rover Workspace

Development workspace for the Earth Rover Challenge Track 01 Urban GPS navigation project. The system follows a hybrid autonomy architecture: learned perception and decision inputs support explicit navigation, control, safety, recovery, logging, replay, and SDK integration.

## Repository Structure

```text
IROS2026/
|-- earth-rovers-sdk/              # FastAPI SDK service and rover endpoints
|-- earth-rover-hybrid-autonomy/   # Navigation, control, safety, replay, and training code
`-- info/                           # Sanitized dataset structure notes
```

The two projects share this Git repository but retain independent dependencies and commands. Run commands from the relevant project directory.

## Development Workflow

```text
MacBook development and Git
        |
        | GitHub + Tailscale/SSH
        v
Dell Ubuntu 22.04 + RTX 4070
        |
        | laboratory LAN
        v
NAS datasets, manifests, checkpoints, and logs
```

Source code is committed from the Mac and pulled onto the Dell laptop for GPU training and inference. Raw datasets, credentials, virtual environments, manifests, checkpoints, and generated outputs remain outside Git.

## Clone on Ubuntu

```bash
git clone git@github.com:cmh0728/IROS2026.git
cd IROS2026
```

Create or activate the project environment before installing dependencies. Python 3.10 is the primary Ubuntu target.

```bash
conda create -n earth-rovers python=3.10 -y
conda activate earth-rovers
```

## SDK Setup

```bash
cd earth-rovers-sdk
pip install -r requirements.txt
cp .env.sample .env
```

Configure `.env` locally. Do not commit credentials. On Ubuntu with the Chromium Snap package, use:

```env
CHROME_EXECUTABLE_PATH=/snap/bin/chromium
```

Start the SDK service:

```bash
hypercorn main:app --bind 127.0.0.1:8000
```

Check server availability without issuing rover motion:

```bash
curl -I http://127.0.0.1:8000/docs
```

See `earth-rovers-sdk/README.md` for endpoint and mission configuration details.

## Hybrid Autonomy Setup

```bash
cd earth-rover-hybrid-autonomy
pip install -r requirements.txt
pytest
```

Run the no-motion SDK smoke test before any rover integration:

```bash
python scripts/run_sdk_smoke_test.py \
  --config configs/default.yaml \
  --no-motion
```

The Urban MVP can send live rover commands. Do not run it against hardware without explicit authorization, verified telemetry, and a tested stop procedure.

## Model Development Status

The inspected FrodoBots-2K sample stores camera data as HLS video and control/sensor data as timestamped CSV files. Initial model work is limited to building a deterministic front-frame/control manifest with timestamp validation. ResNet18 training and HLS frame decoding begin only after that manifest passes focused tests and a small ride-level dry run.

Relevant references:

- `earth-rover-hybrid-autonomy/docs/plans/earth_rover_model_development_next_steps_20260715.md`
- `info/frodobots_2k_sample_structure.md`

## Local Data and Secrets

The root `.gitignore` excludes:

- `datasets_2k/` and `datasets_7k/`;
- `.env` files, keys, and certificates;
- virtual environments and Python caches;
- logs, screenshots, and telemetry samples;
- manifests, training outputs, and experiment runs;
- model checkpoints and exported model files.

Keep raw datasets immutable. Store active data on the Dell laptop or NAS, and pass paths through local configuration or command-line arguments rather than committing machine-specific absolute paths.
