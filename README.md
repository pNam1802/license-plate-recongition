# License Plate Recognition

Hệ thống nhận diện biển số xe Việt Nam theo thời gian thực, chạy trên CPU, hỗ trợ webcam, file video và luồng RTSP.

---

## Demo

<!-- Thay đường dẫn bên dưới bằng ảnh/gif của bạn -->

![Demo tổng quan](docs/demo.png)

<table>
  <tr>
    <td align="center"><img src="docs/demo_zone.png" width="420"/><br/><sub>Vẽ Line / Zone</sub></td>
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
- **Đường kẻ ảo (tripwire)** — chỉ OCR khi xe vượt qua đường, hỗ trợ chọn chiều vào
- **Lưu kết quả** — mỗi biển số được lưu ảnh xe, ảnh biển, frame đầy đủ và file thông tin
- **Fallback top-down** — nếu không detect được xe, tự tìm biển số trực tiếp trên frame

---

## Pipeline — Giải thích chi tiết

### Workflow cơ bản

```
Frame từ video/webcam
        │
        ▼
[1] Vehicle Detection      — YOLO phát hiện vị trí xe trong frame
        │
        ▼
[2] ByteTrack              — gán Track ID duy nhất, theo dõi xuyên suốt
        │
        ▼
[3] Line / Zone Gate       — chỉ xử lý xe đã vượt đường kẻ ảo
        │
        ▼
[4] Plate Detection        — YOLO chuyên dụng crop vùng biển số
        │
        ▼
[5] Perspective Warp       — làm phẳng biển số nghiêng
        │
        ▼
[6] PaddleOCR              — đọc ký tự từ ảnh biển số
        │
        ▼
[7] Voting + Validation    — tổng hợp nhiều lần đọc, chốt kết quả
        │
        ▼
data/results/car-<BIENSỐ>/
```

---

### Các kỹ thuật bên trong

#### Frame Skipping

Video thực tế chạy 25–30 fps nhưng các frame liên tiếp gần như giống hệt nhau — xe chỉ di chuyển vài pixel. Chạy full pipeline mỗi frame tốn CPU mà không thêm thông tin.

**Giải pháp:** Xử lý 1 trong mỗi `--skip N` frames (mặc định N=2). Các frame bỏ qua vẫn được hiển thị để video không giật.

---

#### ByteTrack — Theo dõi xe qua nhiều frame

YOLO chỉ cho biết "frame này có xe ở đây". Nó không biết xe ở frame trước và xe ở frame này là cùng một chiếc hay không.

**ByteTrack** giải quyết bằng cách:
- Gán mỗi xe một **Track ID** duy nhất và duy trì ID đó qua nhiều frame
- Dùng Kalman Filter để dự đoán vị trí xe ở frame tiếp theo
- Dùng Hungarian Algorithm để ghép detection mới với track cũ dựa trên IoU (độ chồng lấp bounding box)

**Tại sao cần:** Nếu không có tracking, mỗi frame ta sẽ thấy "một xe mới" thay vì "cùng chiếc xe". Voting và dedup sẽ không hoạt động được.

**Nhược điểm:** ByteTrack đôi khi đổi ID khi xe bị che khuất hoặc thoát khỏi frame rồi vào lại → hệ thống có cơ chế **dedup 300 frame** để tránh lưu trùng cùng một biển số.

---

#### State Machine — DETECTING → SAMPLING → DONE

Mỗi xe (mỗi Track ID) đi qua 3 trạng thái:

```
DETECTING   — xe mới xuất hiện, đang chờ vượt đường kẻ
     │
     │ (vượt line)
     ▼
SAMPLING    — đang thu thập kết quả OCR qua nhiều frame
     │
     │ (đủ votes hoặc hết max_samples)
     ▼
DONE        — đã chốt biển số, dừng OCR cho xe này
```

**Tại sao cần:** Không có state machine, hệ thống sẽ OCR liên tục mọi xe ở mọi frame, tốn CPU và có thể chốt sai trên kết quả không ổn định.

---

#### OCR Interval

Dù xe đang ở trạng thái SAMPLING, không phải frame nào cũng chạy OCR. Các frame liên tiếp của cùng một xe nhìn gần như giống nhau — chạy OCR mỗi frame chỉ tốn tài nguyên mà không cho thêm thông tin.

**Giải pháp:** Mỗi xe chỉ chạy OCR tối đa 1 lần mỗi `OCR_INTERVAL` frames (mặc định: 3 frames).

---

#### Đường kẻ ảo (Virtual Tripwire)

Thay vì OCR tất cả xe trong frame, hệ thống chỉ bắt đầu đọc biển khi xe **vượt qua đường kẻ** đã vẽ sẵn.

**Cơ chế phát hiện vượt đường:**
Dùng **cross product** để xác định xe đang ở phía nào của đường:
```
val = (P2.x - P1.x) × (cy - P1.y) - (P2.y - P1.y) × (cx - P1.x)
val > 0 → phía +1
val < 0 → phía -1
```
Khi dấu của `val` thay đổi giữa 2 frame liên tiếp → xe vừa vượt qua đường.

**`entry_side`:** Chỉ đếm xe đi từ phía `+1 → -1` (hoặc `-1 → +1` tùy cấu hình). Xe đi ngược chiều không bị đếm và không bị OCR.

Để vẽ đường và chọn chiều:
```bash
python tools/draw_line.py --source video.mp4
# F = đảo chiều, Enter = lưu
```

---

#### Plate Detection — Lọc false positive

Sau khi crop vùng xe, model `plate-detect.pt` tìm biển số bên trong. Có 2 bộ lọc phụ để loại nhiễu:

1. **Diện tích tối thiểu (`MIN_PLATE_AREA`):** Loại bỏ detection quá nhỏ, thường là nhiễu hoặc phản chiếu.

2. **Tỉ lệ khung hình (`W/H ≥ 1.5`):** Biển số xe luôn nằm ngang (rộng hơn cao). Nếu detection có hình gần vuông hoặc cao hơn rộng → đó là đèn xe, logo hoặc vật khác, không phải biển số.

---

#### Perspective Warp — Làm phẳng biển số nghiêng

Camera thực tế không bao giờ nhìn thẳng vuông góc vào biển số. Ảnh biển số thường bị nghiêng hoặc méo theo phối cảnh.

**Giải pháp:** Dùng **4 góc** của bounding box biển số để tính ma trận biến đổi phối cảnh (`getPerspectiveTransform`), rồi áp dụng `warpPerspective` để "kéo thẳng" biển số về hình chữ nhật chuẩn trước khi đưa vào OCR.

**Tại sao quan trọng:** Chữ bị nghiêng làm PaddleOCR đọc sai hoặc bỏ sót ký tự.

---

#### PaddleOCR

Model PaddleX `en_PP-OCRv5_mobile_rec` đọc text từ ảnh biển số đã làm phẳng.

Với biển số **2 dòng** (xe máy Việt Nam), OCR đọc ra 2 dòng text riêng — hệ thống ghép 2 dòng lại thành 1 chuỗi.

**Chuẩn hóa OCR output:**
- Loại bỏ ký tự không phải ASCII và không phải chữ/số (loại bỏ chữ Hán, ký tự đặc biệt)
- Chuyển thành chữ HOA
- Thử sửa lỗi phổ biến: chữ `L` bị nhận nhầm thành `4` ở vị trí số

---

#### Confidence-Weighted Voting

Thay vì lấy kết quả OCR đầu tiên (dễ sai), hệ thống thu thập nhiều lần đọc rồi **bỏ phiếu**:

```
Mỗi kết quả OCR hợp lệ → được tích lũy vào bảng điểm
Điểm của mỗi candidate = tổng confidence của tất cả lần đọc ra kết quả đó
```

Ví dụ:
```
Frame 10: "29K10425" conf=0.91  → điểm 29K10425 = 0.91
Frame 13: "29K10425" conf=0.88  → điểm 29K10425 = 1.79
Frame 16: "29K1O425" conf=0.72  → điểm 29K1O425 = 0.72  (O thay K — regex loại)
Frame 19: "29K10425" conf=0.94  → điểm 29K10425 = 2.73  ← thắng
```

**Điều kiện chốt:** candidate thắng phải đạt tối thiểu `min_votes` lần đọc VÀ chiếm ít nhất `vote_ratio` tổng số votes hợp lệ.

---

#### Regex Validation — Lọc biển số không hợp lệ

Trước khi đưa vào voting, mỗi chuỗi OCR được kiểm tra theo định dạng biển số Việt Nam:

| Loại | Ví dụ | Pattern |
|------|-------|---------|
| Ô tô tiêu chuẩn | `29K10425` | `\d{2}[A-Z]{1,2}\d{4,5}` |
| Ô tô có dấu | `29A-123.45` | `\d{2}[A-Z]{1,2}[-.]?\d{3}[.-]?\d{2}` |
| Xe máy | `29B112345` | `\d{2}[A-Z]\d{1,2}\d{4,5}` |
| Biển 4 số cũ | `29A1234` | `\d{2}[A-Z]{1,2}\d{4}` |

Ngoài ra: 2 số đầu (mã tỉnh) không thể là `00–09`.

Dùng `--no-validate` để tắt regex khi test biển nước ngoài.

---

#### Dedup — Tránh lưu trùng

ByteTrack có thể đổi ID khi xe bị che khuất rồi xuất hiện lại. Nếu không có dedup, cùng một chiếc xe có thể bị lưu nhiều lần dưới các ID khác nhau.

**Giải pháp:** Theo dõi `{plate_text → frame_idx lần chốt cuối}`. Nếu cùng chuỗi biển số được chốt trong vòng 300 frame → bỏ qua lưu file (vẫn hiển thị trên màn hình).

---

#### Fallback Top-Down Camera

Với camera nhìn từ trên xuống (overhead), xe thường không bị nhận dạng tốt bởi model xe thông thường vì góc nhìn khác hoàn toàn.

**Giải pháp:** Nếu YOLO không detect được xe nào trong frame → thử tìm biển số trực tiếp trên toàn frame bằng `plate-detect.pt`. Nếu có → dùng vùng biển số làm "bounding box xe" và chạy OCR thẳng.

---

## Cấu trúc thư mục

```
license-plate-recognition/
├── main.py                    # Entry point — pipeline thời gian thực
│
├── pipeline/
│   ├── vehicle_detector.py    # Stage 1+2: YOLO + ByteTrack
│   ├── plate_reader.py        # Stage 3+4+5: Plate detect + Warp + OCR
│   ├── vote_logic.py          # Stage 6+7: Voting + Regex validation
│   └── zone_filter.py         # LineFilter / ZoneFilter
│
├── tools/
│   ├── draw_line.py           # Vẽ đường kẻ ảo + chọn chiều vào
│   └── draw_zone.py           # Vẽ zone đa giác
│
├── models/
│   ├── yolo26n.pt             # Model phát hiện xe
│   └── plate-detect.pt        # Model phát hiện biển số
│
└── data/
    ├── line.json              # Đường kẻ ảo đã vẽ
    ├── zone.json              # Zone đa giác (tùy chọn)
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
pip install paddlepaddle paddleocr
pip install ultralytics huggingface_hub
pip install opencv-python --force-reinstall
pip install numpy
```

> **GPU:** Thay `paddlepaddle` bằng `paddlepaddle-gpu` và truyền `--device cuda` khi chạy.

---

## Sử dụng

### 1. Vẽ đường kẻ ảo (khuyến nghị)

```bash
python tools/draw_line.py --source video.mp4
# hoặc webcam:
python tools/draw_line.py --source 0
```

| Phím | Chức năng |
|------|-----------|
| Click trái | Đặt điểm (tối đa 2 điểm) |
| Click phải | Xóa điểm vừa đặt |
| `F` | Đảo chiều vào (flip entry direction) |
| `Space` | Chuyển sang frame tiếp theo |
| `R` | Reset |
| `Enter` / `C` | Lưu vào `data/line.json` |
| `Q` | Thoát không lưu |

Mũi tên xanh lá trên màn hình chỉ chiều xe được đếm. Bấm `F` để đảo chiều.

### 2. Chạy pipeline nhận diện

```bash
# Webcam
python main.py --source 0

# File video
python main.py --source video.mp4

# RTSP stream
python main.py --source rtsp://admin:admin@192.168.1.100:554/stream1

# Biển số nước ngoài (bỏ qua regex VN)
python main.py --source video.mp4 --no-validate

# Không lưu video output
python main.py --source video.mp4 --save-img False
```

| Phím | Chức năng |
|------|-----------|
| `Q` | Thoát |
| `P` | Tạm dừng / Tiếp tục |
| `F` | Fullscreen |
| `S` | Lưu frame hiện tại |
| `R` | Reset tracker, counter, kết quả |

---

## Tham số dòng lệnh

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--source` | `0` | Webcam, file video hoặc RTSP URL |
| `--skip` | `2` | Xử lý 1 trong mỗi N frames |
| `--conf` | `0.35` | Ngưỡng confidence vehicle detection |
| `--min-area` | `4000` | Diện tích xe tối thiểu (px²) |
| `--min-votes` | `3` | Số lần đọc tối thiểu để chốt biển số |
| `--vote-ratio` | `0.55` | Tỉ lệ đồng thuận để chốt |
| `--device` | `cpu` | Thiết bị: `cpu` hoặc `cuda` |
| `--no-validate` | tắt | Bỏ qua regex VN — dùng khi test biển nước ngoài |
| `--save-img` | `True` | Lưu video output vào `data/results/` |

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
