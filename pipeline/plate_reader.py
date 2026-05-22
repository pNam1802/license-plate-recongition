"""
Stage 3 + Stage 4 + Stage 5: Plate Detection, Perspective Transform & OCR
==========================================================================
- Stage 3: Crop vùng xe → plate-detect.pt → bbox biển số
- Stage 4: Dùng OpenCV contour tìm 4 góc biển số → warpPerspective
- Stage 5: PaddleOCR đọc ký tự từ ảnh đã làm phẳng
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pipeline.vote_logic import normalize_plate, is_valid_vn_plate

# ---------------------------------------------------------------------------
# OCR-failed save directory
# ---------------------------------------------------------------------------

_FAIL_DIR = "data/ocr-failed"
os.makedirs(_FAIL_DIR, exist_ok=True)

_fail_seq = 0   # global sequence number for unique filenames

def _save_ocr_failed(plate_crop: np.ndarray, track_id: int, info: dict) -> str:
    global _fail_seq
    _fail_seq += 1
    stem = f"{_fail_seq:05d}_id{track_id}"
    cv2.imwrite(os.path.join(_FAIL_DIR, f"{stem}.jpg"), plate_crop)
    with open(os.path.join(_FAIL_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Track ID:   {track_id}\n")
        f.write(f"Thời gian:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    return stem


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PLATE_MODEL_PATH = "models/plate-detect.pt"
PLATE_CONF       = 0.25          # Ngưỡng confidence phát hiện biển số
PLATE_OUT_W      = 300           # Chiều rộng output sau warpPerspective
PLATE_OUT_H      = 100           # Chiều cao output sau warpPerspective
MIN_PLATE_AREA   = 500           # Diện tích biển số tối thiểu (px²)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PlateResult:
    text:         str
    conf:         float
    plate_box:    Tuple[int, int, int, int]
    warped_img:   Optional[np.ndarray] = field(default=None, repr=False)
    vehicle_crop: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Helpers: Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Resize + CLAHE biển số để PaddleOCR recognition hoạt động tốt.
    CLAHE cải thiện độ tương phản giúp đọc ký tự series (A, B, ...) chính xác hơn.
    """
    h, w = img.shape[:2]
    # Scale up to ≥64px height
    if h < 64:
        scale = 64.0 / h
        img = cv2.resize(img, (max(64, int(w * scale)), 64),
                         interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    if w < 120:
        img = cv2.resize(img, (120, h), interpolation=cv2.INTER_CUBIC)

    # CLAHE trên kênh L (lightness) để tăng tương phản chữ
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return img


# ---------------------------------------------------------------------------
# Helpers: Perspective Transform (Phương án A — contour-based)
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: tổng nhỏ nhất
    rect[2] = pts[np.argmax(s)]   # bottom-right: tổng lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: hiệu nhỏ nhất
    rect[3] = pts[np.argmax(diff)]  # bottom-left: hiệu lớn nhất
    return rect


def _find_plate_corners(plate_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Tìm 4 góc biển số từ ảnh crop bằng contour detection.

    Returns:
        numpy array (4, 2) các điểm góc đã sắp xếp, hoặc None nếu không tìm được.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop
    # Blur nhẹ để giảm nhiễu trước khi threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Lấy contour lớn nhất
    largest = max(contours, key=cv2.contourArea)
    peri    = cv2.arcLength(largest, True)

    # Xấp xỉ thành tứ giác
    for eps_factor in [0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(largest, eps_factor * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return _order_points(pts)

    return None


def _warp_plate(plate_crop: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([
        [0,           0          ],
        [PLATE_OUT_W - 1, 0          ],
        [PLATE_OUT_W - 1, PLATE_OUT_H - 1],
        [0,           PLATE_OUT_H - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(plate_crop, M, (PLATE_OUT_W, PLATE_OUT_H))


# ---------------------------------------------------------------------------
# OCR Character Correction
# ---------------------------------------------------------------------------

# Ký tự thường bị nhầm bởi PaddleOCR trên biển số VN (font đặc biệt)
# Chỉ áp dụng tại các vị trí PHẢI là chữ số (mã tỉnh & số hiệu cuối)
_DIGIT_FIX = {
    'Z': '2',   # Z ↔ 2 (rất phổ biến với font biển số VN)
    'I': '1',   # I ↔ 1
    'L': '1',   # L ↔ 1 (cũng có thể là 4, thử 1 trước)
    'O': '0',   # O ↔ 0
    'Q': '0',   # Q ↔ 0
    'S': '5',   # S ↔ 5
    'G': '6',   # G ↔ 6
    'D': '0',   # D ↔ 0
    'B': '8',   # B ↔ 8 (chỉ dùng ở vị trí số, không dùng ở chuỗi ký tự)
}

def _fix_plate_chars(text: str) -> str:
    """
    Sửa lỗi OCR tại các vị trí chắc chắn là chữ số trong biển số VN:
    - 2 ký tự đầu: mã tỉnh (luôn là số)
    - 4-5 ký tự cuối: số hiệu xe (luôn là số)
    Không sửa phần ký tự series ở giữa để tránh làm hỏng chữ hợp lệ (B, S, ...).
    """
    if len(text) < 5:
        return text
    chars = list(text)
    # Mã tỉnh (pos 0-1): phải là chữ số
    for i in range(min(2, len(chars))):
        chars[i] = _DIGIT_FIX.get(chars[i], chars[i])
    # Số hiệu xe (4-5 ký tự cuối): phải là chữ số
    for i in range(max(2, len(chars) - 5), len(chars)):
        chars[i] = _DIGIT_FIX.get(chars[i], chars[i])
    return ''.join(chars)


def _fix_plate_chars_l4(text: str) -> str:
    """Biến thể: thử L→4 thay vì L→1 (một số font in 4 giống L)."""
    fix = {**_DIGIT_FIX, 'L': '4'}
    if len(text) < 5:
        return text
    chars = list(text)
    for i in range(min(2, len(chars))):
        chars[i] = fix.get(chars[i], chars[i])
    for i in range(max(2, len(chars) - 5), len(chars)):
        chars[i] = fix.get(chars[i], chars[i])
    return ''.join(chars)


# ---------------------------------------------------------------------------
# PlateReader
# ---------------------------------------------------------------------------

class PlateReader:
    """
    Đọc biển số xe từ ảnh crop của xe.

    Pipeline nội bộ:
        crop_xe → YOLO detect biển số → warpPerspective → PaddleOCR
    """

    def __init__(
        self,
        plate_model_path: str = PLATE_MODEL_PATH,
        plate_conf: float     = PLATE_CONF,
        device: str           = "cpu",
    ):
        from ultralytics import YOLO

        if not os.path.exists(plate_model_path):
            raise FileNotFoundError(
                f"[PlateReader] Không tìm thấy model: {plate_model_path}"
            )

        self.plate_model = YOLO(plate_model_path)
        self.plate_conf  = plate_conf
        self.device      = device

        print(f"[PlateReader] Plate model: {plate_model_path}")

        # Eager-load recognition model ngay khi khởi tạo
        print("[PlateReader] Đang tải OCR recognition model...")
        try:
            import os as _os
            _os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddlex import create_model as _create_model
            # Dùng trực tiếp recognition model (bỏ qua detection step)
            # vì plate đã được YOLO crop sẵn rồi
            self._ocr = _create_model("en_PP-OCRv5_mobile_rec")
            import paddleocr as _poc
            print(f"[PlateReader] PaddleOCR {_poc.__version__} / paddlex rec model đã sẵn sàng.")
        except Exception as e:
            raise ImportError(
                f"Không load được OCR model: {e}\n\n"
                "Cài đặt: pip install paddlepaddle paddleocr --upgrade\n"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_plates_on_frame(self, frame: np.ndarray) -> list:
        """
        Fallback cho camera góc nhìn từ trên (top-down): detect + track biển số
        trực tiếp trên toàn bộ frame mà không cần vehicle detection.

        Returns list of (VehicleDetection, plate_box) pairs.
        IDs offset thêm 10000 để không trùng với vehicle track IDs.
        """
        from pipeline.vehicle_detector import VehicleDetection

        results = self.plate_model.track(
            frame,
            tracker  = "bytetrack.yaml",
            persist  = True,
            conf     = 0.35,
            verbose  = False,
        )
        if not results or results[0].boxes is None:
            return []

        fh, fw = frame.shape[:2]
        items = []
        for box in results[0].boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)

            det = VehicleDetection(
                track_id   = int(box.id[0]) + 10000,
                class_name = "vehicle",
                box        = (x1, y1, x2, y2),   # original plate bbox
                conf       = float(box.conf[0]),
            )
            items.append(det)
        return items

    def read_direct_plate(
        self,
        frame: np.ndarray,
        plate_box: Tuple[int, int, int, int],
        track_id: int = -1,
    ) -> Optional[PlateResult]:
        """Plate-direct mode (camera top-down): bbox biển đã biết, bỏ qua detect."""
        x1, y1, x2, y2 = plate_box
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        # Context crop quanh biển để lưu làm ảnh xe
        pad = max((x2 - x1), (y2 - y1))
        ctx_crop = frame[max(0, y1-pad):min(fh, y2+pad),
                         max(0, x1-pad):min(fw, x2+pad)]

        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None

        ph, pw = plate_crop.shape[:2]
        ratio  = ph / pw if pw > 0 else 0
        two_line = ratio > 0.55
        suffix = "  → 2-line OCR" if two_line else ""
        print(f"  [LICENSE-PLATE] ID {track_id}: {pw}x{ph}px  H/W={ratio:.2f}{suffix}")

        text, ocr_conf = self._ocr_two_line(plate_crop) if two_line else self._ocr_plate(plate_crop)
        print(f"  [OCR-RESULT]    ID {track_id}: '{text}'  conf={ocr_conf:.2f}")
        if not is_valid_vn_plate(normalize_plate(text or "")):
            stem = _save_ocr_failed(plate_crop, track_id, {
                "OCR text":   text or "Unknown",
                "Kich thuoc": f"{pw}x{ph}px",
                "H/W ratio":  f"{ratio:.2f}",
                "2-line OCR": two_line,
                "OCR conf":   f"{ocr_conf:.2f}",
            })
            print(f"  [OCR-FAIL]      → {_FAIL_DIR}/{stem}.jpg")

        return PlateResult(
            text         = text,
            conf         = ocr_conf,
            plate_box    = (x1, y1, x2, y2),
            warped_img   = plate_crop.copy(),
            vehicle_crop = ctx_crop.copy(),
        )

    def read(
        self,
        frame: np.ndarray,
        vehicle_box: Tuple[int, int, int, int],
        track_id: int = -1,
    ) -> Optional[PlateResult]:
        """Normal mode: crop xe → detect biển → OCR."""
        vx1, vy1, vx2, vy2 = vehicle_box
        fh, fw = frame.shape[:2]
        vx1 = max(0, vx1);  vy1 = max(0, vy1)
        vx2 = min(fw, vx2); vy2 = min(fh, vy2)

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            return None

        plate_result = self._detect_plate(vehicle_crop)
        if plate_result is None:
            print(f"  [LICENSE-PLATE] ID {track_id}: không tìm thấy biển trong crop xe")
            return None

        px1, py1, px2, py2, plate_conf = plate_result
        plate_crop = vehicle_crop[py1:py2, px1:px2]
        if plate_crop.size == 0:
            return None

        ph, pw = plate_crop.shape[:2]
        ratio    = ph / pw if pw > 0 else 0
        two_line = ratio > 0.55
        suffix   = "  → 2-line OCR" if two_line else ""
        print(f"  [LICENSE-PLATE] ID {track_id}: conf={plate_conf:.2f}  {pw}x{ph}px  H/W={ratio:.2f}{suffix}")

        text, ocr_conf = self._ocr_two_line(plate_crop) if two_line else self._ocr_plate(plate_crop)
        abs_box = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
        print(f"  [OCR-RESULT]    ID {track_id}: '{text}'  conf={ocr_conf:.2f}")
        if not is_valid_vn_plate(normalize_plate(text or "")):
            stem = _save_ocr_failed(plate_crop, track_id, {
                "OCR text":       text or "Unknown",
                "Kich thuoc":     f"{pw}x{ph}px",
                "H/W ratio":      f"{ratio:.2f}",
                "Plate det conf": f"{plate_conf:.2f}",
                "2-line OCR":     two_line,
                "OCR conf":       f"{ocr_conf:.2f}",
            })
            print(f"  [OCR-FAIL]      → {_FAIL_DIR}/{stem}.jpg")

        return PlateResult(
            text         = text,
            conf         = ocr_conf,
            plate_box    = abs_box,
            warped_img   = plate_crop.copy(),
            vehicle_crop = vehicle_crop.copy(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_plate(
        self, vehicle_crop: np.ndarray
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Phát hiện biển số trong ảnh crop của xe.

        Returns:
            (x1, y1, x2, y2, conf) trong toạ độ vehicle_crop, hoặc None.
        """
        results = self.plate_model(
            vehicle_crop,
            conf    = self.plate_conf,
            device  = self.device,
            verbose = False,
        )
        if not results or results[0].boxes is None:
            return None

        best = None
        best_area = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            area  = (x2 - x1) * (y2 - y1)
            
            # Bỏ qua MIN_PLATE_AREA để debug xem có tìm thấy biển quá nhỏ không
            # if area < MIN_PLATE_AREA:
            #     continue
            
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2, conf)

        return best

    def _ocr_raw(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Chạy recognition trên một ảnh, trả về (text_alphanumeric, conf).
        Không áp dụng _fix_plate_chars — dùng nội bộ bởi _ocr_plate và _ocr_two_line.
        """
        processed = _preprocess_for_ocr(img)
        try:
            results = list(self._ocr.predict(processed))
        except Exception as e:
            print(f"  [OCR-ERR] predict exception: {e}")
            return "Unknown", 0.0

        if not results:
            return "Unknown", 0.0

        texts, confs = [], []
        for r in results:
            if not isinstance(r, dict):
                continue
            t = r.get("rec_text", "")
            c = r.get("rec_score", 0.0)
            if t:
                clean = "".join(ch for ch in t if ch.isalnum()).upper().strip()
                if clean:
                    texts.append(clean)
                    confs.append(float(c))

        if not texts:
            return "Unknown", 0.0

        return "".join(texts), sum(confs) / len(confs)

    def _ocr_plate(self, img: np.ndarray) -> Tuple[str, float]:
        """OCR biển 1 dòng: recognition + sửa ký tự số bị nhầm."""
        text, conf = self._ocr_raw(img)
        if text != "Unknown":
            text = _fix_plate_chars(text)
        return text, conf

    def _ocr_two_line(self, img: np.ndarray) -> Tuple[str, float]:
        """
        OCR biển 2 dòng (xe máy): chia đôi theo chiều cao, OCR từng nửa riêng,
        ghép kết quả.

        Biển xe máy VN 2 dòng:
          Dòng 1: mã tỉnh + ký tự series + nhóm  (vd: "29K1")
          Dòng 2: số hiệu xe                      (vd: "12345")
          Ghép lại:                                 "29K112345"
        """
        h, w = img.shape[:2]
        mid = h // 2

        top_text, top_conf = self._ocr_raw(img[:mid, :])
        bot_text, bot_conf = self._ocr_raw(img[mid:, :])
        print(f"    [2LINE] top='{top_text}'({top_conf:.2f})  bot='{bot_text}'({bot_conf:.2f})")

        parts = [t for t in (top_text, bot_text) if t and t != "Unknown"]
        if not parts:
            return "Unknown", 0.0

        combined = "".join(parts)
        mean_conf = (top_conf + bot_conf) / 2
        combined = _fix_plate_chars(combined)
        return combined, mean_conf
