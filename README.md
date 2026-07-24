# Earth Rover Hybrid Autonomy Workspace

Earth Rover Challenge Track 01 Urban GPS Navigation을 위한 하이브리드 자율주행 연구 저장소입니다. 학습 모델은 전방 영상에서 주행 가능 영역을 추정하고, 임무 방향 결정과 실제 제어는 GPS waypoint, local planner, controller, safety supervisor가 담당합니다.

> 현재 모델 출력은 offline 연구·검수용입니다. SDK, planner, controller 또는 live rover 명령과 연결된 상태가 아닙니다.

## 시스템 구성

```text
전방 카메라
  -> traversability segmentation
  -> mask / confidence
                            ┐
GPS waypoint + IMU         ├-> local planner -> controller -> safety -> SDK
  -> 목표 방향과 거리       ┘
```

End-to-end learned control이 아니라 학습 기반 perception과 고전 navigation/control을 분리한 구조입니다. 센서 누락, stale telemetry, 통신 실패는 정지 또는 보수적인 동작으로 처리하는 것을 기본 원칙으로 합니다.

## 저장소 구조

```text
IROS2026/
├── earth-rovers-sdk/              # FastAPI/Hypercorn 기반 SDK 서비스
├── earth-rover-hybrid-autonomy/   # navigation, planning, control, safety, replay, training
└── info/                           # 민감정보를 제거한 dataset 구조 조사 자료
```

두 하위 프로젝트는 의존성과 실행 명령이 다릅니다. 명령은 해당 프로젝트 디렉터리에서 실행하세요.

## 개발 환경

```text
MacBook
  코드 작성, Git, 문서 검토
       |
       | GitHub + SSH
       v
Dell Ubuntu 22.04 / RTX 4070
  dataset 처리, 테스트, CUDA 추론, 학습
       |
       v
NAS 또는 Dell 로컬 저장소
  원본 dataset, checkpoint, experiment, review bundle
```

Python 3.10을 기준으로 합니다. dataset, checkpoint, 영상, 로그와 `.env`는 Git에 포함하지 않습니다.

## 설치

저장소를 clone한 뒤 프로젝트별로 설치합니다.

```bash
git clone <REPOSITORY_SSH_URL>
cd IROS2026
```

### Hybrid autonomy와 모델 개발

```bash
cd earth-rover-hybrid-autonomy
python3 -m pip install -r requirements.txt
```

Linux x86_64에서는 `requirements.txt`가 확인된 CUDA 11.8용 PyTorch wheel과 SegFormer runtime을 설치합니다. Mac에서는 CUDA PyTorch 항목이 environment marker에 의해 제외됩니다. H.264 검수 영상 생성에는 시스템 `ffmpeg`가 필요합니다.

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264
```

Focused test:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider -q
```

### SDK

```bash
cd earth-rovers-sdk
python3 -m pip install -r requirements.txt
cp .env.sample .env
hypercorn main:app --bind 127.0.0.1:8000
```

Ubuntu Chromium Snap 경로는 `.env`에 다음과 같이 설정할 수 있습니다.

```env
CHROME_EXECUTABLE_PATH=/snap/bin/chromium
```

`.env`의 API key, token, mission 정보는 절대 commit하지 마세요. 실제 rover 연결 전에는 no-motion smoke test를 사용합니다.

```bash
cd ../earth-rover-hybrid-autonomy
python3 scripts/run_sdk_smoke_test.py --config configs/default.yaml --no-motion
```

## 모델 개발 현황

### ResNet18 action baseline

전방 단일 이미지에서 `FORWARD`, `LEFT`, `RIGHT`, `STOP`, `REVERSE`를 분류하는 초기 pipeline baseline입니다. Manifest, HLS lazy decoding, ride-level split, checkpoint, metric, 추론 영상 경로 검증에 사용했으며 production controller로 사용하지 않습니다. 단일 이미지에는 route intent가 보이지 않는다는 구조적 한계가 있습니다.

### Traversability SegFormer-B0 v2

현재 검수 중인 최신 모델입니다.

- Backbone: `nvidia/segformer-b0-finetuned-ade-512-512`
- 입력: 화면비를 유지해 letterbox한 512 크기 전방 RGB frame
- 출력: `ON_ROAD`, `OFF_ROAD`, `OBSTACLE`의 per-pixel logits와 confidence
- 학습 데이터: 승인된 v1 120장과 신규 수동 annotation 33장을 병합한 v2 dataset
- 초기화: 승인된 SegFormer-B0 v1 best checkpoint에서 낮은 learning rate로 fine-tuning
- 평가: 기존 v1 고정 evaluation split과 신규 ride-group holdout을 분리
- 최신 checkpoint 기본 경로:
  `$HOME/datasets/experiments/traversability_segformer_b0_v2/full_training/segformer_b0_best.pt`

최근 `output_rides_0`, `output_rides_1`, `output_rides_2`의 서로 다른 ride를 이용한 offline 연속 영상 추론과 사람 검수를 수행했습니다. 전체적인 mask 품질은 양호했지만, 이는 안전성 검증이나 실차 검증을 의미하지 않습니다. 연석, 거친 포장, 자갈, 그림자, 역광과 새로운 장소에서의 false-traversable 오류는 계속 확인해야 합니다. 정량 metric과 실행 환경은 Git 외부 experiment report 및 `review_manifest.json`을 기준으로 확인합니다.

## Label Contract

Dataset에 저장하는 단일 채널 PNG mask:

| Source ID | Class | 의미 | CVAT 색상 |
|---:|---|---|---|
| 0 | `IGNORE` | 불확실하거나 학습하지 않는 영역 | 검정 |
| 1 | `ON_ROAD` | 주행 가능한 포장 지면 | 초록 |
| 2 | `OFF_ROAD` | 주행 가능한 흙, 자갈, 짧은 잔디 | 파랑 |
| 3 | `OBSTACLE` | 연석, 계단, 벽, 사람, 차량 등 | 빨강 |

학습 시에는 `IGNORE -> 255`, `ON_ROAD -> 0`, `OFF_ROAD -> 1`, `OBSTACLE -> 2`로 변환하며 `ignore_index=255`를 사용합니다.

연속 영상 검수 도구는 가시성을 위해 `OFF_ROAD`를 **노란색**으로 표시합니다. 이는 visualization만의 색상이며 dataset class ID와 CVAT 색상은 변경되지 않습니다.

## FrodoBots Dataset 사용

기본 dataset 위치:

```text
$HOME/datasets/
├── output_rides_0/
├── output_rides_1/
└── output_rides_2/
```

각 root는 다음 형식을 기대합니다.

```text
ride_<ID>_*/
├── front_camera_timestamps_<ID>.csv
└── recordings/
    ├── *uid_s_1000*video.m3u8
    └── *.ts
```

`uid_s_1000`만 전방 카메라로 사용하고 `uid_s_1001`은 제외합니다. 원본 dataset은 수정하거나 전체 추출하지 않습니다.

### v2 모델로 연속 영상 검사

Dell에서 다음 스크립트를 실행하면 dataset마다 서로 다른 ride 5개와 ride당 60초 구간을 deterministic하게 선택합니다. 결과는 original, overlay, prediction mask로 구성된 QuickTime 호환 H.264 영상입니다.

```bash
cd earth-rover-hybrid-autonomy
./scripts/run_traversability_video_review_v2.sh
```

기본 결과:

```text
$HOME/datasets/review_bundles/traversability_video_review_v2/
├── review_manifest.json
├── output_rides_0/
│   ├── traversability_review.mp4
│   └── review_manifest.json
├── output_rides_1/
└── output_rides_2/
```

개별 dataset만 검사:

```bash
DATASETS=0 OUTPUT_DIR="$HOME/datasets/review_bundles/traversability_video_review_v2_rides0" ./scripts/run_traversability_video_review_v2.sh
```

저신뢰도 pixel을 검게 표시하는 별도 검수 결과:

```bash
LOW_CONFIDENCE_THRESHOLD=0.55 OUTPUT_DIR="$HOME/datasets/review_bundles/traversability_video_review_v2_confidence" ./scripts/run_traversability_video_review_v2.sh
```

다른 동일 형식 dataset을 직접 지정할 수도 있습니다.

```bash
python3 training/run_traversability_video_review_v2.py \
  --checkpoint "$HOME/datasets/experiments/traversability_segformer_b0_v2/full_training/segformer_b0_best.pt" \
  --config configs/traversability_segformer_b0_v2.yaml \
  --dataset-root-0 /path/to/output_rides_custom \
  --datasets 0 \
  --output-dir "$HOME/datasets/review_bundles/custom_review" \
  --require-cuda
```

기존 output은 자동으로 덮어쓰지 않습니다. 의도적으로 교체할 때만 CLI의 `--overwrite` 또는 shell script의 `OVERWRITE=true`를 사용하세요.

## Dataset과 모델 재현

승인된 annotation에서 v2 dataset 생성:

```bash
./scripts/build_traversability_dataset_v2.sh
```

생성된 split report에서 ride leakage와 class 분포를 확인한 후에만 fine-tuning과 v1/v2 비교를 실행합니다.

```bash
./scripts/run_traversability_segformer_b0_v2.sh
```

자세한 단계와 이전 Phase 1~3 검증 기록은 `earth-rover-hybrid-autonomy/training/README.md` 및 `docs/`를 참고하세요.

## 안전 및 저장소 관리

- 학습 모델은 mission direction을 결정하거나 직접 rover 명령을 생성하지 않습니다.
- live rover 동작은 현재 작업별 명시적 승인과 중단 절차 없이는 실행하지 않습니다.
- raw dataset, normalized mask, checkpoint, Hugging Face cache, MP4, manifest와 experiment 결과는 Git 외부에 둡니다.
- `.env`, token, private key, Tailscale 주소와 사용자별 절대 경로를 commit하지 않습니다.
- 큰 검수 영상은 Git commit 대신 GitHub Release asset 또는 별도 artifact storage를 사용합니다.
