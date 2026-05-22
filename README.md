# License Plate Recognition

Hệ thống nhận diện biển số xe Việt Nam theo thời gian thực, chạy trên CPU, hỗ trợ webcam, file video và luồng RTSP.

---

## Demo

<!-- Thay đường dẫn bên dưới bằng ảnh/gif của bạn -->

![Demo tổng quan](docs/demo.png)

<table>
  <tr>
    <td align="center"><img src="docs/demo_zone.png" width="420"/><br/><sub>Vẽ Zone of Interest</sub></td>
    <td align="center"><img src="docs/demo_result.png" width="420"/><br/><sub>Kết quả nhận diện thời gian thực</sub></td>
  </tr>
</table>

---

## Tính năng

- **Phát hiện xe** — YOLOv8 (`yolo26n.pt`) nhận diện ô tô, xe máy, xe tải, xe buýt
- **Theo dõi đa xe** — ByteTrack gán ID duy nhất, duy trì xuyên suốt video
- **Phát hiện biển số** — model YOLO chuyên dụng (`plate-detect.pt`) crop chính xác vùng biển số
- **Làm phẳng góc nghiêng** — `warpPerspective` căn chỉnh biển số trước khi đọc
- **OCR** — PaddleOCR đọc ký tự, hỗ trợ biển số 1 dòng và 2 dòng
- **Voting & xác thực** — Confidence-Weighted Voting + Regex biển số Việt Nam, tránh chốt sai
- **Zone of Interest** — chỉ OCR khi xe vào vùng đã khoanh, hỗ trợ **2 zone độc lập** (ví dụ 2 làn đường)
- **Lưu kết quả** — mỗi biển số được lưu ảnh xe, ảnh biển, frame đầy đủ và file thông tin
- **Fallback top-down** — nếu không detect được xe, tự tìm biển số trực tiếp trên frame (camera nhìn từ trên xuống)

---

## Cấu trúc thư mục

```
license-plate-recognition/
├── main.py                    # Entry point — pipeline thời gian thực
├── plate_detector.py          # Nhận diện biển số trên ảnh đơn (legacy)
│
├── pipeline/
│   ├── vehicle_detector.py    # Stage 1+2: YOLO + ByteTrack
│   ├── plate_reader.py        # Stage 3+4+5: Plate detect + Warp + OCR
│   ├── vote_logic.py          # Stage 6+7: Voting + Regex validation
│   └── zone_filter.py         # ZoneFilter / MultiZoneFilter
│
├── tools/
│   └── draw_zone.py           # Vẽ 1 hoặc 2 zone bằng chuột
│
├── models/
│   ├── yolo26n.pt             # Model phát hiện xe
│   └── plate-detect.pt        # Model phát hiện biển số
│
└── data/
    ├── zones.json             # Zone đã vẽ (2 zones)
    ├── zone.json              # Zone cũ (1 zone, vẫn tương thích)
    └── results/               # Kết quả nhận diện đã chốt
        └── car-<BIENSỐ>/
            ├── plate.jpg
            ├── vehicle.jpg
            ├── frame.jpg
            └── info.txt
```

---

## Cài đặt

**Yêu cầu:** Python 3.10+

```bash
# 1. Cài dependencies theo thứ tự
pip install paddlepaddle paddleocr
pip install ultralytics huggingface_hub
pip install opencv-python --force-reinstall
pip install numpy
```

> **GPU:** Thay `paddlepaddle` bằng `paddlepaddle-gpu` và truyền `--device cuda` khi chạy.

---

## Sử dụng

### 1. Vẽ Zone of Interest (khuyến nghị)

Mở video lên, khoanh vùng trên frame nơi biển số hiện rõ nhất — pipeline chỉ OCR xe đi qua vùng này.

```bash
python tools/draw_zone.py --source video.mp4
# hoặc webcam:
python tools/draw_zone.py --source 0
```

| Phím | Chức năng |
|------|-----------|
| Click trái | Thêm điểm vào zone đang chỉnh |
| Click phải | Xóa điểm vừa thêm |
| `Tab` | Chuyển đổi giữa Zone 1 và Zone 2 |
| `R` | Reset zone đang active |
| `E` | Reset tất cả zones |
| `Space` | Chuyển sang frame tiếp theo |
| `Enter` / `C` | Lưu và thoát (cần ≥ 3 điểm mỗi zone) |
| `Q` | Thoát không lưu |

Kết quả lưu vào `data/zones.json`.

### 2. Chạy pipeline nhận diện

```bash
# Webcam
python main.py --source 0

# File video
python main.py --source video.mp4

# RTSP stream
python main.py --source rtsp://admin:admin@192.168.1.100:554/stream1

# Tùy chỉnh tham số
python main.py --source video.mp4 --skip 3 --conf 0.45 --min-votes 3
```

| Phím | Chức năng |
|------|-----------|
| `Q` | Thoát |
| `P` | Tạm dừng / Tiếp tục |
| `F` | Fullscreen |
| `S` | Lưu frame hiện tại |
| `R` | Reset tracker |

---

## Tham số dòng lệnh

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--source` | `0` | Webcam, file video hoặc RTSP URL |
| `--skip` | `2` | Xử lý 1 trong mỗi N frames |
| `--conf` | `0.40` | Ngưỡng confidence vehicle detection |
| `--min-area` | `4000` | Diện tích xe tối thiểu (px²) |
| `--min-votes` | `2` | Số lần đọc tối thiểu để chốt biển số |
| `--vote-ratio` | `0.50` | Tỉ lệ đồng thuận để chốt |
| `--device` | `cpu` | Thiết bị: `cpu` hoặc `cuda` |

---

## Pipeline xử lý

```
Frame
  │
  ▼
[1] Vehicle Detection — YOLO phát hiện xe, lọc theo confidence & diện tích
  │
  ▼
[2] ByteTrack — gán Track ID duy nhất, duy trì qua nhiều frames
  │
  ▼
[3] Zone Filter — chỉ xử lý xe nằm trong Zone of Interest
  │
  ▼
[4] Plate Detection — crop vùng xe, phát hiện biển số bằng YOLO chuyên dụng
  │
  ▼
[5] Perspective Warp — tìm 4 góc biển số, làm phẳng góc nghiêng
  │
  ▼
[6] PaddleOCR — đọc ký tự từ ảnh đã làm phẳng
  │
  ▼
[7] Voting + Validation — Regex VN + Confidence-Weighted Voting → chốt kết quả
  │
  ▼
data/results/car-<BIENSỐ>/
```

---

## Kết quả đầu ra

Mỗi biển số được chốt sẽ tạo một thư mục `data/results/car-<BIENSỐ>/`:

```
data/results/car-29K10425/
├── plate.jpg      # Crop biển số đã làm phẳng
├── vehicle.jpg    # Crop vùng xe
├── frame.jpg      # Frame đầy đủ tại thời điểm chốt
└── info.txt       # Track ID, confidence, frame, thời gian
```

---

## Ghi chú

- **ByteTrack** đôi khi đổi ID khi xe bị che khuất — hệ thống có cơ chế dedup 300 frame để tránh lưu trùng cùng một biển số.
- **Top-down camera:** Nếu YOLO không detect được xe (góc nhìn từ trên xuống), pipeline tự động tìm biển số trực tiếp trên toàn frame.
- **Không có zone:** Nếu chưa vẽ zone, hệ thống xử lý toàn bộ frame.
