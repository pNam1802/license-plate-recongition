# Giải thích Pipeline Nhận Diện Biển Số

---

## Mục lục

1. [Luồng xử lý tổng quan](#1-luồng-xử-lý-tổng-quan)
2. [Frame Skipping](#2-frame-skipping)
3. [Vehicle Detection — YOLO](#3-vehicle-detection--yolo)
4. [ByteTrack — Theo dõi xe](#4-bytetrack--theo-dõi-xe)
5. [State Machine](#5-state-machine)
6. [Virtual Tripwire — Đường kẻ ảo](#6-virtual-tripwire--đường-kẻ-ảo)
7. [Plate Detection — Tìm biển số](#7-plate-detection--tìm-biển-số)
8. [Perspective Warp — Làm phẳng](#8-perspective-warp--làm-phẳng)
9. [PaddleOCR — Đọc ký tự](#9-paddleocr--đọc-ký-tự)
10. [Voting — Bỏ phiếu chốt kết quả](#10-voting--bỏ-phiếu-chốt-kết-quả)
11. [Dedup — Chống lưu trùng](#11-dedup--chống-lưu-trùng)
12. [Fallback Top-Down](#12-fallback-top-down)
13. [Các tham số quan trọng](#13-các-tham-số-quan-trọng)

---

## 1. Luồng xử lý tổng quan

Hình dung pipeline như một dây chuyền sản xuất. Mỗi frame video đi qua từng công đoạn theo thứ tự sau:

```
┌─────────────────────────────────────────────────────────────┐
│                        1 FRAME MỚI                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                    Frame Skipping?
                    (xử lý frame này không?)
                             │ có
                             ▼
                ┌────────────────────────┐
                │   YOLO Vehicle Detect  │  → tìm tất cả xe trong frame
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │       ByteTrack        │  → gán Track ID duy nhất cho mỗi xe
                └────────────┬───────────┘
                             │
                   (với mỗi xe riêng lẻ)
                             │
                             ▼
                ┌────────────────────────┐
                │   Kiểm tra Line Cross  │  → xe có vượt đường kẻ không?
                └────────────┬───────────┘
                             │ có (lần đầu tiên)
                             ▼
                ┌────────────────────────┐
                │   Plate Detection      │  → YOLO tìm biển số bên trong xe
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │   Perspective Warp     │  → làm phẳng ảnh biển nghiêng
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │      PaddleOCR         │  → đọc ký tự ra chuỗi text
                └────────────┬───────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  Validate + Voting     │  → kiểm tra định dạng, tích lũy votes
                └────────────┬───────────┘
                             │
                   đủ điều kiện chốt?
                             │ có
                             ▼
                ┌────────────────────────┐
                │    Lưu kết quả         │  → plate.jpg, vehicle.jpg, info.txt
                └────────────────────────┘
```

---

## 2. Frame Skipping

### Vấn đề
Video chạy ở 25–30 fps. Hai frame liên tiếp cách nhau khoảng 33ms — xe chỉ di chuyển vài pixel. Chạy toàn bộ pipeline (YOLO + OCR) cho mỗi frame sẽ tốn CPU mà không có thêm thông tin mới.

### Giải pháp
Chỉ xử lý 1 frame trong mỗi N frame (mặc định N=2):

```
Frame 1 → xử lý ✓
Frame 2 → bỏ qua, hiển thị lại frame 1
Frame 3 → xử lý ✓
Frame 4 → bỏ qua, hiển thị lại frame 3
...
```

Các frame bị bỏ qua vẫn được **hiển thị lên màn hình** (frame cũ) để video không bị giật. Chỉ riêng việc chạy detection là bị skip.

### Ảnh hưởng
- N=1: xử lý mọi frame — chính xác nhất nhưng chậm
- N=2: nhanh gấp đôi, độ trễ ~66ms — đủ dùng thực tế
- N=3: nhanh hơn nữa nhưng có thể bỏ sót xe đi nhanh

---

## 3. Vehicle Detection — YOLO

### YOLO là gì?
YOLO (You Only Look Once) là một mạng neural network nhận đầu vào là ảnh và đầu ra là danh sách các vật thể tìm được, mỗi vật thể gồm:
- **Bounding box**: tọa độ (x1, y1, x2, y2) vùng chứa vật thể
- **Class**: loại vật thể (car, motorcycle, truck, bus)
- **Confidence**: xác suất model tin đây đúng là vật thể đó (0.0 → 1.0)

### Bộ lọc đầu ra
Sau khi YOLO trả về kết quả, có 2 bộ lọc phụ:

**1. Confidence threshold (--conf, mặc định 0.35)**
Loại bỏ detection có confidence thấp — model không chắc đây là xe.

**2. Diện tích tối thiểu (min_area, mặc định 4000 px²)**
Loại bỏ xe quá nhỏ trong frame. Xe nhỏ thường ở xa, biển số quá mờ để đọc được.

**3. NMS — Non-Maximum Suppression (iou=0.35)**
YOLO đôi khi phát hiện cùng một xe 2 lần với 2 bounding box chồng lên nhau. NMS tự động giữ lại box có confidence cao nhất và loại box còn lại nếu chúng chồng lấp nhau quá 35%.

---

## 4. ByteTrack — Theo dõi xe

### Vấn đề YOLO không giải quyết được
YOLO chỉ biết "trong **frame này** có xe ở vị trí X". Nó không biết xe đó có phải cùng chiếc xe với frame trước không.

Không có tracking → mỗi frame ta thấy "một xe mới" → không thể tích lũy kết quả OCR qua nhiều frame → voting không hoạt động.

### ByteTrack giải quyết như thế nào?

**Bước 1 — Kalman Filter dự đoán**
Dựa vào vận tốc và hướng di chuyển của xe từ frame trước, Kalman Filter **dự đoán** xe sẽ ở đâu trong frame hiện tại.

**Bước 2 — Hungarian Algorithm ghép đôi**
So sánh các detection mới từ YOLO với các dự đoán:
- Tính IoU (độ chồng lấp) giữa mỗi detection mới với mỗi track cũ
- Hungarian Algorithm tìm cách ghép đôi sao cho tổng IoU là lớn nhất
- Detection ghép được với track cũ → giữ nguyên Track ID
- Detection không ghép được → tạo Track ID mới

**Bước 3 — Cập nhật Kalman**
Kalman Filter cập nhật lại mô hình chuyển động dựa trên vị trí thực tế vừa đo được.

### Ví dụ cụ thể
```
Frame 10: YOLO thấy xe ở (100,200) → ByteTrack tạo ID=5
Frame 11: YOLO thấy xe ở (108,203) → ByteTrack nhận ra là ID=5 (di chuyển 8px)
Frame 12: Xe bị khuất sau cột → YOLO không thấy
           ByteTrack giữ ID=5 trong bộ nhớ, dự đoán vị trí
Frame 13: YOLO thấy xe ở (124,209) → ByteTrack nhận ra lại là ID=5
```

### Nhược điểm của ByteTrack
- Khi xe bị khuất quá lâu (> track_timeout frames), ByteTrack "quên" track đó
- Khi xe xuất hiện lại, nó được gán **ID mới** → hệ thống coi là xe khác
- Đây là lý do tại sao cần cơ chế **Dedup** (xem phần 11)

---

## 5. State Machine

### Tại sao cần state machine?
Nếu không có state machine, pipeline sẽ:
- OCR mọi xe ở mọi frame → tốn CPU
- Không biết xe nào đã được đọc xong, xe nào chưa
- Không thể kiểm soát việc bắt đầu/dừng OCR

### Ba trạng thái

```
DETECTING ──────────────────────────────► SAMPLING ──────────► DONE
           (xe vượt line, hoặc không có line)         (đủ votes)
```

**DETECTING** — Xe mới xuất hiện trong frame. Hệ thống đang theo dõi vị trí nhưng **chưa chạy OCR**. Xe ở trạng thái này cho đến khi vượt đường kẻ ảo.

**SAMPLING** — Xe đã vượt line. Hệ thống bắt đầu chạy OCR, tích lũy kết quả vào bảng voting. Mỗi xe chỉ chạy OCR tối đa 1 lần mỗi `OCR_INTERVAL` frames để tránh lãng phí.

**DONE** — Đã chốt được biển số (đủ votes) hoặc đã thử quá `max_samples` lần mà không chốt được. Hệ thống hoàn toàn **bỏ qua** xe này, không OCR thêm.

### Tại sao quan trọng
Không có DONE state → một xe đã chốt biển số vẫn tiếp tục bị OCR mỗi frame cho đến khi ra khỏi màn hình → lãng phí CPU.

---

## 6. Virtual Tripwire — Đường kẻ ảo

### Mục đích
Thay vì OCR tất cả xe trong frame (bao gồm xe xa, xe mờ, xe chưa vào đúng góc), chỉ bắt đầu OCR khi xe **thực sự đi vào vùng quan sát** — tức là vượt qua đường kẻ ta đã vẽ.

### Cơ chế — Cross Product

Cho đường kẻ từ điểm P1 đến P2. Với mỗi xe, tính:

```
val = (P2.x - P1.x) × (cy - P1.y) - (P2.y - P1.y) × (cx - P1.x)
```

Trong đó `(cx, cy)` là tâm bounding box của xe.

- `val > 0` → xe đang ở **phía +1** (bên phải/bên dưới vector P1→P2)
- `val < 0` → xe đang ở **phía -1** (bên trái/bên trên)
- `val = 0` → xe đang nằm đúng trên đường

**Phát hiện vượt đường:** Lưu lại phía của xe ở frame trước (`prev_side`). Nếu frame này `side != prev_side` → xe vừa vượt qua đường.

### entry_side — Chọn chiều vào

Không phải mọi hướng vượt đường đều được đếm. Ví dụ camera ở cổng vào bãi đỗ xe — chỉ muốn đếm xe **đi vào**, không đếm xe đi ra.

```
entry_side = +1  → chỉ đếm xe đi từ phía +1 sang phía -1
entry_side = -1  → chỉ đếm xe đi từ phía -1 sang phía +1
entry_side =  0  → đếm cả 2 chiều
```

Trong `draw_line.py`, mũi tên xanh lá chỉ chiều xe được đếm. Bấm `F` để đảo chiều.

---

## 7. Plate Detection — Tìm biển số

### Quy trình

```
Ảnh xe (crop từ frame)
        │
        ▼
plate-detect.pt (YOLO chuyên dụng)
        │
        ▼
Danh sách bounding box biển số trong ảnh xe
        │
        ├── Bộ lọc 1: Diện tích < MIN_PLATE_AREA (500 px²) → loại
        │
        └── Bộ lọc 2: W/H < 1.5 → loại (đèn xe, logo, vật tròn)
```

### Tại sao lọc theo tỉ lệ W/H?

Biển số xe **luôn nằm ngang** — chiều rộng lớn hơn chiều cao đáng kể. Tỉ lệ W/H của biển số VN thường từ 2.0 đến 4.0.

Đèn pha, logo xe, gương chiếu hậu có tỉ lệ gần vuông (W/H ≈ 1.0) hoặc cao hơn rộng. Bộ lọc `W/H ≥ 1.5` loại bỏ những thứ này.

---

## 8. Perspective Warp — Làm phẳng

### Vấn đề
Camera không bao giờ nhìn thẳng vuông góc vào biển số. Biển số trong thực tế thường trông như thế này:

```
Ảnh biển số thực tế (nghiêng, méo):      Ảnh sau warp (phẳng):

  ╱‾‾‾‾‾‾‾‾‾‾‾‾╲                         ┌────────────────┐
 ╱  29K · 10425  ╲           →            │  29K · 10425   │
╱________________╲                        └────────────────┘
```

Chữ bị nghiêng làm PaddleOCR khó đọc và dễ nhầm ký tự.

### Cơ chế

1. Lấy **4 góc** của bounding box biển số
2. Định nghĩa **4 góc đích** — hình chữ nhật phẳng với kích thước chuẩn
3. `getPerspectiveTransform` tính ma trận biến đổi 3×3
4. `warpPerspective` áp dụng ma trận lên ảnh biển số → kết quả là ảnh phẳng

---

## 9. PaddleOCR — Đọc ký tự

### Quy trình

```
Ảnh biển phẳng
      │
      ▼
en_PP-OCRv5_mobile_rec (model nhận dạng ký tự)
      │
      ▼
Raw text: "29K 10425\n"  conf=0.91
      │
      ├── Strip ký tự không phải ASCII/alnum
      │   (loại ký tự Trung, tiếng Thái, ký tự đặc biệt)
      │
      ├── Ghép 2 dòng (biển xe máy 2 dòng)
      │   "29K" + "10425" → "29K10425"
      │
      └── Uppercase → "29K10425"
```

### Sửa lỗi L → 4
Font chữ một số biển số VN in chữ `4` trông rất giống `L`. PaddleOCR đôi khi đọc ra `L` thay vì `4`.

Hệ thống thử thay `L` bằng `4` ở các vị trí số trong chuỗi, rồi kiểm tra lại regex — nếu hợp lệ thì dùng bản đã sửa.

---

## 10. Voting — Bỏ phiếu chốt kết quả

Đây là phần thường gây nhầm lẫn nhất. Giải thích chi tiết bằng ví dụ thực tế.

### Tại sao không lấy kết quả OCR đầu tiên?

Vì OCR **không hoàn hảo**. Cùng một biển số, mỗi frame cho ra kết quả khác nhau:

```
Frame 10: "29K10425"   (ảnh rõ)
Frame 13: "29K1O425"   (nhầm 0 → O)
Frame 16: "29K10425"   (ảnh rõ)
Frame 19: "Z9K10425"   (nhầm 2 → Z, ảnh tối)
Frame 22: "29K10425"   (ảnh rõ)
```

Nếu lấy kết quả đầu tiên → dễ ra sai. Nếu lấy đa số → chính xác hơn nhiều.

### Confidence-Weighted Voting — Cụ thể từng bước

**Bước 1 — Nhận kết quả OCR mới**

Mỗi lần OCR trả về một chuỗi và một confidence score. Confidence là mức độ chắc chắn của PaddleOCR (0.0 → 1.0).

```
OCR result: text="29K10425"  conf=0.91
```

**Bước 2 — Validate**

Kiểm tra chuỗi có đúng định dạng biển số VN không (xem phần Regex). Nếu không hợp lệ → **bỏ qua**, không thêm vào history.

```
"29K10425" → regex match ✓ → thêm vào history
"Z9K10425" → regex fail  ✗ → bỏ qua
```

**Bước 3 — Tích lũy vào bảng điểm**

Mỗi candidate (chuỗi biển số) được tích lũy điểm = tổng confidence của tất cả lần đọc ra nó:

```
Sau frame 10: score["29K10425"] = 0.91  count["29K10425"] = 1
Sau frame 13: "29K1O425" → regex fail → bỏ qua
Sau frame 16: score["29K10425"] = 0.91 + 0.88 = 1.79   count = 2
Sau frame 19: "Z9K10425" → regex fail → bỏ qua
Sau frame 22: score["29K10425"] = 1.79 + 0.94 = 2.73   count = 3
```

**Bước 4 — Kiểm tra điều kiện chốt**

Tại mỗi frame, sau khi thêm vote mới, kiểm tra:

```
Tổng số vote hợp lệ = 3  (chỉ tính những lần regex pass)
Candidate tốt nhất  = "29K10425"  count = 3

Điều kiện 1: count >= min_votes (3)?    3 >= 3 ✓
Điều kiện 2: count / total >= vote_ratio (0.55)?   3/3 = 1.0 >= 0.55 ✓

→ CHỐT: "29K10425"
```

**Ví dụ khi chưa đủ (vote_ratio chưa đạt):**

```
Tổng = 10 votes hợp lệ
"29K10425": count=4   score=3.20
"29K10426": count=6   score=4.80  ← nhiều hơn nhưng confidence thấp

count tốt nhất là "29K10426" có 6 votes → count/total = 6/10 = 0.6 ✓
→ chốt "29K10426"
```

Nhưng nếu:
```
"29K10425": count=4   score=3.20
"2BK10425": count=3   score=2.10
"29K10425": count=3   score=2.80  (cộng thêm 3 lần nữa)

count tốt nhất là "29K10425" có 7 votes → count/total = 7/10 = 0.7 ✓
→ chốt "29K10425"
```

**Điều gì xảy ra khi không chốt được?**

Nếu sau `max_samples` lần OCR (mặc định 30 lần) mà không candidate nào đủ votes → xe chuyển sang DONE nhưng **không có biển số**, không lưu file. Pipeline bỏ qua xe này.

### Tóm tắt tại sao dùng Confidence-Weighted thay vì đơn giản đếm số lần xuất hiện?

Ví dụ: "29K10425" xuất hiện 3 lần với confidence 0.92, 0.90, 0.91 (ảnh rõ nét, ban ngày).
Và "29K1O425" xuất hiện 5 lần với confidence 0.45, 0.42, 0.38, 0.40, 0.44 (ảnh tối, mờ).

Đếm đơn giản → "29K1O425" thắng (5 > 3) → sai.
Confidence-weighted → "29K10425" thắng (0.91×3=2.73 > 0.42×5=2.10) → đúng.

---

## 11. Dedup — Chống lưu trùng

### Vấn đề
ByteTrack đôi khi đổi ID (xe bị che khuất rồi xuất hiện lại). Khi đó cùng một chiếc xe có thể:
- Frame 1–100: Track ID = 5, biển "29K10425" → lưu file ✓
- Frame 101–110: Xe đi khuất sau trụ điện
- Frame 111–200: ByteTrack gán ID mới = 12, biển "29K10425" → lưu file nữa → TRÙNG

### Giải pháp

Theo dõi dictionary `{plate_text → frame_idx lần chốt cuối}`.

Mỗi lần sắp lưu:
```
"29K10425" đã được chốt ở frame 85
Hiện tại frame 120
Khoảng cách: 120 - 85 = 35 frames < DEDUP_FRAMES (300)
→ bỏ qua lưu file (vẫn hiển thị trên màn hình)
```

Nếu cùng biển số xuất hiện sau 300 frames → coi là xe khác, lưu bình thường.

---

## 12. Fallback Top-Down

### Khi nào xảy ra?
Camera nhìn từ trên xuống (overhead camera trên cổng, trên giao lộ). Với góc này, nhìn thấy nóc xe, không nhìn thấy đầu/đuôi xe → model xe thông thường không nhận ra.

### Giải pháp
```python
vehicles = vehicle_detector.track(frame)   # → rỗng (không tìm thấy xe nào)

if not vehicles:
    vehicles = plate_reader.track_plates_on_frame(frame)  # tìm biển trực tiếp
    plate_direct = True
```

Chạy `plate-detect.pt` thẳng trên toàn frame thay vì trên crop xe. Vùng biển tìm được được dùng làm "bounding box xe giả". Từ đó pipeline tiếp tục bình thường (OCR trên crop biển đó).

---

## 13. Các tham số quan trọng

| Tham số | File | Mặc định | Ý nghĩa | Tăng lên khi | Giảm xuống khi |
|---|---|---|---|---|---|
| `SKIP_FRAMES` | main.py | 2 | Xử lý 1/N frame | CPU yếu | Cần độ chính xác cao |
| `VEHICLE_CONF` | main.py | 0.35 | Ngưỡng detect xe | Nhiều false positive | Bỏ sót xe thật |
| `MIN_AREA` | main.py | 4000 | Diện tích xe tối thiểu | Xe xa bị lấy | Xe nhỏ bị bỏ |
| `OCR_INTERVAL` | main.py | 3 | Frame giữa 2 lần OCR | CPU yếu | Xe đi nhanh |
| `MIN_VOTES` | main.py | 3 | Số lần đọc để chốt | Nhiều false positive | Xe đi qua nhanh |
| `VOTE_RATIO` | main.py | 0.55 | Tỉ lệ đồng thuận | Cần chắc chắn hơn | Kết quả OCR hay thay đổi |
| `MAX_SAMPLES` | main.py | 30 | Số lần OCR tối đa | Xe đứng lâu | Tốn CPU |
| `TRACK_TIMEOUT` | main.py | 90 | Frame trước khi xóa track | Xe hay bị khuất | Xe thoáng qua |
| `DEDUP_FRAMES` | main.py | 300 | Khoảng cách chống trùng | Nhiều xe giống biển | Xe qua lại nhanh |
| `PLATE_CONF` | plate_reader.py | 0.20 | Ngưỡng detect biển | Nhiều false positive | Bỏ sót biển số |
| `MIN_PLATE_AREA` | plate_reader.py | 500 | Diện tích biển tối thiểu | Nhiễu nhỏ | Biển số nhỏ trong frame |

---

## Sơ đồ quyết định trong 1 frame

```
frame mới
    │
    ├─ frame_idx % skip != 0? → hiển thị frame cũ, bỏ qua
    │
    └─ xử lý frame này:
          │
          ├─ YOLO detect xe → [xe1, xe2, xe3, ...]
          │
          ├─ cleanup(): xóa track cũ (không xuất hiện > 90 frame, chưa chốt)
          │
          └─ với mỗi xe:
                │
                ├─ update_seen()  ← ghi nhận xe còn sống
                │
                ├─ check_crossing()  ← tính cross product
                │
                ├─ state == DONE? → vẽ bbox, bỏ qua
                │
                ├─ state == DETECTING + chưa vượt line? → vẽ bbox, bỏ qua
                │
                ├─ state == DETECTING + vừa vượt line → đếm +1, chuyển SAMPLING
                │
                └─ state == SAMPLING:
                      │
                      ├─ should_run_ocr()? → không → bỏ qua
                      │
                      └─ có:
                            │
                            ├─ plate_detect → crop biển → warp → OCR
                            │
                            ├─ validate (regex) → không hợp lệ → ghi nhận "miss"
                            │
                            ├─ hợp lệ → thêm vào history, tính vote
                            │
                            └─ đủ votes? → DONE, lưu file
```
