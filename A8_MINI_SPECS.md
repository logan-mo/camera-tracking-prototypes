# SIYI A8 mini — Specifications

Sourced from SIYI's [spec page](https://siyi.biz/en/product/tri-axis-single-camera-gimbal/a8-mini/spec/)
and the [A8 mini User Manual v1.10](https://res.siyi.biz/oss/other/2026/06/15/A8_mini_User_Manual_v1_10_563cde30.pdf)
(retrieved 2026-09-23). Where the two disagree, both values are given. Values
marked *derived* are computed here, not published by SIYI.

## Camera

| Spec | Value |
|---|---|
| Image sensor | Sony 1/1.7", 8 MP effective |
| Lens | Fixed focal length, F2.8 |
| Equivalent focal length | 21 mm (35 mm-equivalent) |
| FOV | Diagonal 93°, Horizontal 81° (vertical not published) |
| Digital zoom | Up to 6x at 720p, 5.5x at 1080p, 3.5x at 2K, **none at 4K** |
| White balance | Auto |
| Still photo | Single shot, JPG, same resolution as video recording |

## Video

| Spec | Value |
|---|---|
| **Live stream (Ethernet/RTSP)** | Max 1080p ("Stream Resolution" setting; 4K is recording-only) |
| Recording resolutions (all 25 fps) | 4K, 2K (2560×1440), 1080p (1920×1080), 720p (1280×720) |
| 4K width | 4096×2160 per spec page; 3840×2160 per manual |
| Recording codec / bitrate | H.265; 20 Mbps (4K/2K), 15 Mbps (1080p/720p) |
| Recording container | MP4 on microSD (Class 10, ≤256 GB, FAT32/exFAT) |
| Video outputs | Ethernet (RTSP), Micro-HDMI, CVBS (analog, via the Ethernet connector) |

The published specs don't give the stream's frame rate, codec or bitrate.
Read them from the camera with `client.get_encoding_params(...)` in `siyi_sdk`.

## Gimbal

| Spec | Value |
|---|---|
| Stabilization | 3-axis (yaw, pitch, roll) |
| Angular vibration | ±0.01° |
| Controllable pitch | −135° ~ +45° per spec page; −90° ~ +25° per manual v1.10 |
| Controllable yaw | −160° ~ +160° per spec page; −135° ~ +135° per manual v1.10 |
| Roll | −30° ~ +30° |
| Modes | Lock, Follow, FPV |
| Max rotation speed | Not published |

The manual is the more recent document. Treat its narrower ranges as safe
limits until the actual range is confirmed on this unit.

## Electrical & physical

| Spec | Value |
|---|---|
| Voltage | 11 ~ 25.2 V (3S–6S). Units made before June 2023 may not tolerate 25.2 V |
| Power | 5 W average, 12 W peak |
| Dimensions | 55 × 55 × 70 mm |
| Weight | 95 g |
| Ingress protection | IP4X |
| Operating temp | −10 ~ 50 °C |

## Network & control

| Spec | Value |
|---|---|
| Camera IP | `192.168.144.25` (static, no DHCP) |
| PC IP | Any free `192.168.144.x/24`, e.g. `.100` |
| SDK control | SIYI SDK protocol, UDP or TCP port `37260` |
| Other control inputs | UART (SIYI SDK or MAVLink via flight controller), S.Bus |
| RTSP (legacy A8 mini path) | `rtsp://192.168.144.25:8554/main.264` |
| RTSP (newer firmware, per manual v1.10) | Main `rtsp://192.168.144.25:8554/video1`, sub `.../video2` |

If `main.264` doesn't open, try `video1`. Which path works depends on the
firmware version.

## Derived values for tracking (*derived*)

These use a pinhole model with HFOV = 81°, so they are approximate. They also
assume the stream keeps the full horizontal FOV. Calibrate with a checkerboard
before relying on them.

| Quantity | Value |
|---|---|
| VFOV at 16:9 | ≈ 51.3° |
| Focal length in pixels, 1920 wide | fx ≈ 1124 px |
| Focal length in pixels, 1280 wide | fx ≈ 749 px |
| Angle per pixel at image centre (1080p) | ≈ 0.051° |

The published 93° diagonal only matches 81° horizontal on a **4:3** frame
(4:3 gives ≈93.7°; 16:9 would give ≈88.8°). So SIYI's FOV figures most likely
describe the full 4:3 sensor, and a 16:9 stream may be cropped or distorted.
This is another reason to calibrate.
