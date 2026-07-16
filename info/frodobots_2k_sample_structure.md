# FrodoBots-2K Raw Ride Sample Structure

## Scope and Validation

- Sample: `datasets_2k/output_rides_0/`
- Analysis date: 2026-07-14
- State: already extracted directory, not an archive
- Size: 22,534,506,722 bytes (about 21 GiB)
- Contents: 231 rides, 20,263 files

This is a read-only structural checkpoint. The analysis included a complete file inventory, CSV parsing, HLS reference checks, timestamp statistics, and representative `ffprobe` checks. It did not decode every media segment or run training. Revalidate these findings if the sample changes.

## Directory Layout

```text
output_rides_0/
  ride_<ride_id>_<datetime>/
    control_data_<ride_id>.csv
    front_camera_timestamps_<ride_id>.csv
    rear_camera_timestamps_<ride_id>.csv
    gps_data_<ride_id>.csv
    imu_data_<ride_id>.csv
    speaker_audio_timestamps_<ride_id>.csv
    mic_audio_timestamps_<ride_id>.csv       # 30 of 231 rides only
    recordings/
      *uid_s_1000*video.m3u8 / *.ts         # front camera
      *uid_s_1001*video.m3u8 / *.ts         # rear camera
      *uid_s_1001*audio.m3u8 / *.ts         # speaker audio
      *uid_s_<dynamic>*audio.m3u8 / *.ts    # microphone, when present
```

All 231 rides contain the six core CSV files and a `recordings/` directory. There are 1,416 CSV files, 723 HLS playlists, and 18,121 MPEG-TS segments. No empty files were found. Three `.DS_Store` files are non-dataset artifacts.

## Modalities and Schemas

### Cameras

Both camera timestamp files use:

```csv
frame_id,timestamp
```

`frame_id` is zero-based and sequential. `timestamp` is Unix time in seconds. Front and rear timestamps run at approximately 20 Hz (median interval `0.05 s`). Images are not stored as standalone files; frames must be decoded from HLS/MPEG-TS.

| Stream | Codec and shape | Timestamp rows | Rows per ride (min/median/max) |
| --- | --- | ---: | ---: |
| Front | H.264, 1024x576, 20 fps | 1,835,497 | 41 / 3,994 / 71,901 |
| Rear | H.264, 540x360, 20 fps | 1,837,120 | 42 / 3,986 / 71,898 |

The UID-to-camera mapping was confirmed from playlist timing, CSV timestamps, and sampled codec metadata. It is an observed dataset convention, not documented metadata.

### Control and RPM

```csv
linear,angular,rpm_1,rpm_2,rpm_3,rpm_4,timestamp
```

There are 503,230 rows, with 30 / 1,400 / 13,840 rows per ride. `timestamp` is Unix time in seconds; the median positive interval is `0.1 s`, or about 10 Hz. `linear` and `angular` are within `[-1, 1]`. RPM values are present, but wheel order, units, and sign conventions are not documented. RPM must not be treated as measured speed without further validation.

Using the existing `training/datasets/action_labels.py` thresholds, the command distribution is:

| Label | Rows | Share |
| --- | ---: | ---: |
| `STOP` | 163,550 | 32.50% |
| `FORWARD` | 278,814 | 55.40% |
| `LEFT` | 32,177 | 6.39% |
| `RIGHT` | 24,719 | 4.91% |
| `REVERSE` | 3,970 | 0.79% |

These are command-derived labels. The files do not prove whether each command was issued by a human or an autonomous controller.

### GPS

```csv
latitude,longitude,timestamp
```

There are 71,405 rows, with 7 / 179 / 3,193 rows per ride. GPS timestamps are Unix milliseconds and must be divided by 1,000 before alignment. The median positive interval is `1.001 s`. Coordinates are numeric and within legal latitude/longitude ranges; no `(0, 0)` records were found. Ten transitions imply speeds above `30 m/s`, including an extreme jump, so position outliers require filtering.

No mission, route, waypoint, or success metadata was found. A route can only be reconstructed from the GPS trace.

### IMU

```csv
compass,accelerometer,gyroscope,timestamp
```

The three sensor fields contain JSON-encoded arrays whose values are strings. Each inner record has the form `[x, y, z, timestamp_seconds]`. Every outer row contains one compass record, 100 accelerometer records, and one gyroscope record. The outer `timestamp` is Unix milliseconds.

- Accelerometer: approximately 100 Hz, median inner interval `0.01 s`
- Compass and gyroscope: approximately 1 Hz, median inner interval about `1.005 s`
- Outer rows: 66,524 total, 5 / 170 / 2,996 per ride

Use inner timestamps for sensor alignment. Outer timestamps can lag the latest inner sample and show rare delays of tens of seconds. Sensor units, axes, coordinate frame, and calibration are unknown. No explicit yaw, quaternion, or orientation label exists.

### Audio

Speaker timestamps exist in all rides; microphone timestamps exist in 30 rides. Both use `frame_id,timestamp`, with sequential frame IDs and Unix-second timestamps. Sampled streams are AAC mono at 48 kHz, with timestamp steps near `0.02-0.03 s` (consistent with 1,024-sample AAC frames).

Microphone streams contain long discontinuities, up to about `503 s`. The exact microphone/speaker semantics, content consent, and permitted ML use must be checked before audio training.

## HLS Integrity

All 723 playlists contain `#EXT-X-ENDLIST`. Every referenced TS file exists, and no unreferenced TS files were found. Ten representative TS files from early, middle, late, and microphone-containing rides passed `ffprobe`; this does not establish full integrity for all 18,121 segments.

Playlist timelines contain 41 discontinuities greater than `0.1 s` across 20 playlists, primarily in dynamic microphone streams but also in some core streams. Segment and timestamp count mismatches are especially large in a few rides. A loader must treat each discontinuity as a boundary rather than interpolating across it.

## Timestamp Alignment

Normalize all timestamps to Unix seconds before matching. Observed nearest-neighbor alignment quality:

| Pair | Median or threshold result |
| --- | --- |
| Front to rear | median `14 ms`; 99.02% within `25 ms`; max `3.622 s` |
| Front to control | 58.05% within `100 ms`; 60.24% within `200 ms` |
| Front to GPS | 72.53% within `500 ms` |
| Front to IMU outer row | 68.45% within `500 ms` |

The overlap of all key modalities varies widely by ride: median overlap/union is about `31.4%`, while some rides have almost no common interval. Do not assume every camera frame has a usable control or sensor target.

Recommended alignment policy:

1. Use a front-camera frame timestamp as the sample anchor.
2. Decode its frame from the corresponding HLS segment.
3. Accept control only within a documented tolerance, initially `200 ms`; otherwise reject the sample.
4. Match rear camera within `50 ms` when used, and expose the actual delta.
5. Match GPS and IMU independently, retaining age and validity masks.
6. Use IMU inner timestamps, not only the outer CSV timestamp.
7. Never interpolate across a backward timestamp, HLS discontinuity, or long sensor gap.

## Data Quality Findings

- No empty CSVs, malformed numeric rows, missing playlist references, or frame-ID gaps were found.
- Camera timestamp gaps above `75 ms`: 68 front and 1,775 rear; worst gaps are about 18 seconds.
- Control timestamps contain 3,587 duplicates and 10 backward steps.
- IMU inner timestamps contain eight within-ride regressions per sensor and gaps approaching 300 seconds.
- GPS contains implausible position jumps that require speed- or distance-based rejection.
- One RPM channel reaches an isolated value of 356, outside the usual observed range.
- Some commands are non-zero while all RPM values are zero; commands and physical response are not interchangeable.
- Full media decoding, camera calibration, and hardware-ground-truth validation were not performed.

## Usable Inputs and Targets

Stable candidate inputs are front RGB, optional rear RGB, GPS, nested IMU samples, RPM, prior control, and optional audio. Each optional modality should carry a timestamp age and missing-data mask.

The most defensible supervised targets are `linear`/`angular` commands or the derived five-class action label. Possible future targets include filtered future GPS displacement or RPM response, but their lag, units, and outliers require separate validation. The sample does not contain explicit obstacle, traversability, collision, route-completion, or mission-success labels.

## First Baseline Task

A practical offline baseline is front image to coarse action classification (`STOP`, `FORWARD`, `LEFT`, `RIGHT`, `REVERSE`). Its purpose should be dataset and alignment validation, not live rover control. Split by ride rather than by frame to prevent adjacent-frame leakage, account for class imbalance, and reject samples without a sufficiently close control target.

This baseline does not change the hybrid autonomy direction: learned outputs should remain perception or decision inputs behind explicit planning, control, and safety logic unless an architectural change is separately approved.

## Dataset Loader Proposal

No loader was implemented during this analysis. A future loader should:

1. Build an immutable ride and playlist manifest without modifying raw data.
2. Normalize timestamp units and divide rides into monotonic timeline sections.
3. Map camera frame IDs to HLS segments and decode lazily.
4. Align control, rear camera, GPS, and inner IMU records to each front timestamp.
5. Store alignment deltas, freshness masks, and quality flags in the manifest.
6. Reject stale targets, decode failures, discontinuity crossings, and invalid GPS jumps.
7. Create train/validation/test splits at ride level.
8. Keep indexes, caches, decoded frames, and checkpoints outside Git.

Useful manifest fields include `ride_id`, front segment/frame index, normalized timestamp, rear reference/delta, `linear`, `angular`, RPM values, control age, GPS values/age/mask, IMU reference/age/mask, quality flags, and split.

Existing Berkeley-FrodoBots-7K exploration/download scripts do not directly support this raw FrodoBots-2K HLS layout. Reuse `training/datasets/action_labels.py` for label derivation, but do not combine 2K and 7K pipelines until their actual schemas are verified as compatible.

## Unresolved Questions

The directory structure alone cannot determine:

- command provenance and command-to-actuation latency;
- angular sign semantics;
- RPM units, wheel mapping, and sensor meaning;
- IMU units, axes, frame, calibration, and orientation convention;
- camera intrinsics, distortion, and front/rear calibration;
- exact microphone and speaker semantics;
- timezone metadata encoded in ride directory names;
- dataset license, redistribution, privacy, and audio-consent constraints;
- mission goals, checkpoints, route completion, or failure state;
- the cause of streams that outlive sensors or have media/timestamp mismatches;
- integrity of every TS segment without a complete decode pass.

Treat these as unknowns, not defaults, until authoritative metadata or further measurements resolve them.
