"""
Stage 6 + Stage 7: Regex Validation & Voting Logic
====================================================
- State Machine: DETECTING → SAMPLING → DONE cho mỗi track_id
- Lưu lịch sử OCR kết quả per track_id
- Lọc chuỗi không hợp lệ bằng Regex biển số Việt Nam
- Áp dụng Confidence-Weighted Voting để chốt kết quả cuối cùng
- Dọn dẹp track_id không còn xuất hiện sau timeout
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class VehicleState(Enum):
    DETECTING = auto()   # Xe vừa xuất hiện, chờ vào Zone of Interest
    SAMPLING  = auto()   # Xe trong vùng tốt, đang thu thập mẫu OCR
    DONE      = auto()   # Đã chốt biển số — dừng OCR cho xe này


# ---------------------------------------------------------------------------
# Vietnamese License Plate Regex
# ---------------------------------------------------------------------------
# Biển số VN gồm các định dạng phổ biến:
#   Ô tô:   29A-12345  /  29A1-2345  /  29AA-123.45
#   Xe máy: 29-B1-12345  /  29B1-12345
#   Biển 4 số cũ: 29A-1234

_PLATE_PATTERNS = [
    # Ô tô tiêu chuẩn mới (5 số, có thể có dấu chấm): 29A12345 / 29A123.45
    r"^\d{2}[A-HK-Z]{1,2}\d{3,5}$",
    # Ô tô có gạch/chấm: 29A-123.45 / 29AA-12345
    r"^\d{2}[A-HK-Z]{1,2}[-.]?\d{3}[.-]?\d{2}$",
    # Xe máy: 29B112345 / 29-B1-12345
    r"^\d{2}[A-HK-Z]\d{1,2}\d{4,5}$",
    # Biển 4 số cũ: 29A1234
    r"^\d{2}[A-HK-Z]{1,2}\d{4}$",
]

_COMPILED_PATTERNS = [re.compile(p) for p in _PLATE_PATTERNS]

# Biển số không thể bắt đầu bằng 0 ở phần số tỉnh
_INVALID_PROVINCE_START = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09"}


def is_valid_vn_plate(text: str) -> bool:
    """
    Kiểm tra chuỗi có đúng định dạng biển số Việt Nam không.

    Args:
        text: Chuỗi đã được chuẩn hóa uppercase, chỉ chứa chữ và số

    Returns:
        True nếu hợp lệ
    """
    if len(text) < 7 or len(text) > 12:
        return False

    # Kiểm tra 2 số đầu (mã tỉnh)
    if not text[:2].isdigit():
        return False
    if text[:2] in _INVALID_PROVINCE_START:
        return False

    # Match với ít nhất một pattern
    for pattern in _COMPILED_PATTERNS:
        if pattern.match(text):
            return True

    return False


def normalize_plate(text: str) -> str:
    """
    Chuẩn hóa chuỗi OCR: loại bỏ dấu, chuyển uppercase,
    chỉ giữ lại ký tự ASCII chữ và số (loại bỏ chữ Hán, ký tự đặc biệt).
    """
    return "".join(c for c in text.upper() if c.isascii() and c.isalnum())


# ---------------------------------------------------------------------------
# TrackRecord
# ---------------------------------------------------------------------------

@dataclass
class TrackRecord:
    """Lưu toàn bộ trạng thái và lịch sử OCR cho một track_id."""
    state:            VehicleState   = VehicleState.DETECTING
    history:          List[str]      = field(default_factory=list)
    conf_history:     List[float]    = field(default_factory=list)   # confidence per vote
    last_seen_frame:  int            = 0
    last_ocr_frame:   int            = -999   # Frame cuối cùng đã chạy OCR
    sample_count:     int            = 0      # Tổng số lần đã chạy OCR
    confirmed_plate:  Optional[str]  = None   # Kết quả đã được chốt


# ---------------------------------------------------------------------------
# VoteManager
# ---------------------------------------------------------------------------

class VoteManager:
    """
    Quản lý State Machine + Voting kết quả OCR cho nhiều track_id.

    Flow:
        1. Xe xuất hiện → DETECTING
        2. Xe vào Zone → SAMPLING
        3. Mỗi OCR_INTERVAL frame → gọi record_ocr()
        4. Đủ votes → DONE, confirmed_plate được set
        5. Xe rời màn hình → cleanup() xóa record

    Cách dùng:
        vm = VoteManager()

        # Mỗi frame: cập nhật xe đang thấy
        vm.update_seen(track_id=1, frame_idx=10)

        # Khi xe vào Zone of Interest
        vm.transition_to_sampling(track_id=1)

        # Khi đã chạy OCR
        vm.record_ocr(track_id=1, text="30A12345", conf=0.9, frame_idx=15)

        # Kiểm tra kết quả
        plate = vm.get_confirmed(track_id=1)

        # Dọn dẹp
        vm.cleanup(current_frame=100)
    """

    def __init__(
        self,
        min_votes:     int   = 3,     # Số lần tối thiểu để chốt kết quả
        vote_ratio:    float = 0.5,   # Tỉ lệ tối thiểu (xuất hiện / tổng) để chốt
        track_timeout: int   = 90,    # Số frame không xuất hiện thì xóa track
        ocr_interval:  int   = 5,     # Số frame giữa 2 lần OCR cho cùng 1 xe
        max_samples:   int   = 40,    # Số lần OCR tối đa trước khi bỏ qua xe
        validate:      bool  = True,  # False = bỏ qua regex validation (test biển nước ngoài)
    ):
        self.min_votes     = min_votes
        self.vote_ratio    = vote_ratio
        self.track_timeout = track_timeout
        self.ocr_interval  = ocr_interval
        self.max_samples   = max_samples
        self.validate      = validate
        self._records: Dict[int, TrackRecord] = defaultdict(TrackRecord)

    # ------------------------------------------------------------------
    # State Machine API
    # ------------------------------------------------------------------

    def get_state(self, track_id: int) -> VehicleState:
        """Trả về trạng thái hiện tại của xe."""
        return self._records[track_id].state

    def transition_to_sampling(self, track_id: int) -> None:
        """Chuyển xe sang trạng thái SAMPLING (khi vào Zone of Interest)."""
        rec = self._records[track_id]
        if rec.state == VehicleState.DETECTING:
            rec.state = VehicleState.SAMPLING
            print(f"  [STATE] ID {track_id}: DETECTING → SAMPLING")

    def transition_to_done(self, track_id: int) -> None:
        """Chuyển xe sang trạng thái DONE (đã chốt biển số)."""
        rec = self._records[track_id]
        if rec.state != VehicleState.DONE:
            rec.state = VehicleState.DONE
            print(f"  [STATE] ID {track_id}: SAMPLING → DONE")

    # ------------------------------------------------------------------
    # Sampling Control
    # ------------------------------------------------------------------

    def should_run_ocr(self, track_id: int, frame_idx: int) -> bool:
        """
        Kiểm tra xe có được phép chạy OCR ở frame này không.

        Điều kiện:
        - Trạng thái là SAMPLING
        - Đã đủ khoảng cách frame từ lần OCR trước (ocr_interval)
        - Chưa đạt giới hạn số lần thử (max_samples)
        """
        rec = self._records[track_id]
        if rec.state != VehicleState.SAMPLING:
            return False
        if rec.sample_count >= self.max_samples:
            return False
        if (frame_idx - rec.last_ocr_frame) < self.ocr_interval:
            return False
        return True

    def mark_ocr_attempted(self, track_id: int, frame_idx: int) -> None:
        """Ghi nhận đã thử chạy OCR (dù có kết quả hay không)."""
        rec = self._records[track_id]
        rec.last_ocr_frame = frame_idx
        rec.sample_count  += 1

        # Tự động chuyển DONE nếu đã thử quá nhiều lần
        if rec.sample_count >= self.max_samples and rec.state == VehicleState.SAMPLING:
            print(f"  [STATE] ID {track_id}: hết {self.max_samples} lần thử → DONE (không chốt được)")
            rec.state = VehicleState.DONE

    # ------------------------------------------------------------------
    # Voting API
    # ------------------------------------------------------------------

    def add(self, track_id: int, raw_text: str, conf: float, frame_idx: int) -> None:
        """
        Thêm một kết quả OCR mới cho track_id (kèm confidence).

        Args:
            track_id:  ID từ ByteTrack
            raw_text:  Chuỗi OCR thô (chưa normalize)
            conf:      Confidence score của kết quả OCR
            frame_idx: Frame index hiện tại
        """
        record = self._records[track_id]
        record.last_seen_frame = frame_idx

        norm = normalize_plate(raw_text)

        if not self.validate:
            # Chế độ không validate: chấp nhận bất kỳ chuỗi ASCII nào có ≥4 ký tự
            if norm and len(norm) >= 4:
                record.history.append(norm)
                record.conf_history.append(conf)
            else:
                print(f"    [SKIP] ID {track_id}: '{norm}' quá ngắn")
            return

        if norm and is_valid_vn_plate(norm):
            record.history.append(norm)
            record.conf_history.append(conf)
            return

        # Fallback: thử L→4 ở vị trí số (một số font biển số in "4" giống "L")
        chars = list(norm)
        for i in list(range(min(2, len(chars)))) + list(range(max(2, len(chars) - 5), len(chars))):
            if chars[i] == 'L':
                chars[i] = '4'
        norm_l4 = ''.join(chars)
        if norm_l4 != norm and is_valid_vn_plate(norm_l4):
            print(f"    [FIX] ID {track_id}: '{norm}' → '{norm_l4}' (L→4)")
            record.history.append(norm_l4)
            record.conf_history.append(conf)
            return

        print(f"    [SKIP] ID {track_id}: '{norm}' không hợp lệ (len={len(norm)})")

    def update_seen(self, track_id: int, frame_idx: int) -> None:
        """
        Cập nhật last_seen cho track_id dù không có kết quả OCR.
        Dùng để tránh xóa track_id đang active nhưng chưa đọc được biển số.
        """
        self._records[track_id].last_seen_frame = frame_idx

    def get_best(self, track_id: int) -> Optional[str]:
        """
        Trả về biển số tốt nhất theo confidence-weighted voting.

        Cách tính: mỗi candidate nhận điểm = tổng confidence của các lần
        đọc ra kết quả đó. Candidate có tổng điểm cao nhất và đủ ngưỡng
        sẽ được chốt.

        Returns:
            Chuỗi biển số hoặc None nếu chưa đủ dữ liệu / chưa chốt.
        """
        record = self._records.get(track_id)
        if record is None or not record.history:
            return None

        # Tính confidence-weighted score cho từng candidate
        score: Dict[str, float] = {}
        count: Dict[str, int]   = {}
        for plate, conf in zip(record.history, record.conf_history):
            score[plate] = score.get(plate, 0.0) + conf
            count[plate] = count.get(plate, 0)   + 1

        # Lấy candidate có score cao nhất
        best = max(score, key=score.__getitem__)
        best_count = count[best]
        total = len(record.history)

        # Điều kiện chốt: đủ số votes VÀ tỉ lệ đủ cao
        if best_count >= self.min_votes and (best_count / total) >= self.vote_ratio:
            record.confirmed_plate = best
            return best

        return None

    def record_ocr(
        self,
        track_id:  int,
        text:      Optional[str],
        conf:      float,
        frame_idx: int,
    ) -> Optional[str]:
        """
        Tổng hợp: ghi nhận kết quả OCR, thử voting, và trả về biển số
        nếu đã chốt được.

        Args:
            track_id:  ID xe
            text:      Chuỗi OCR (có thể là None/"Unknown" nếu không đọc được)
            conf:      Confidence của OCR
            frame_idx: Frame hiện tại

        Returns:
            Biển số đã chốt, hoặc None.
        """
        self.mark_ocr_attempted(track_id, frame_idx)

        if text and text != "Unknown":
            self.add(track_id, text, conf, frame_idx)
            confirmed = self.get_best(track_id)
            if confirmed:
                self.transition_to_done(track_id)
                return confirmed

        return None

    def get_confirmed(self, track_id: int) -> Optional[str]:
        """Trả về kết quả đã chốt trước đó (không tính lại)."""
        record = self._records.get(track_id)
        if record is None:
            return None
        return record.confirmed_plate

    def get_history_summary(self, track_id: int) -> Dict:
        """Trả về thông tin debug cho một track."""
        record = self._records.get(track_id)
        if record is None:
            return {}
        counter = Counter(record.history)
        return {
            "state":       record.state.name,
            "total_reads": len(record.history),
            "candidates":  counter.most_common(5),
            "confirmed":   record.confirmed_plate,
            "samples":     record.sample_count,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, current_frame: int) -> List[int]:
        """
        Xóa các track_id chưa chốt và không xuất hiện trong `track_timeout` frames.
        Track đã DONE không bị xóa (tránh mất dữ liệu đang hiển thị).

        Returns:
            Danh sách track_id đã bị xóa.
        """
        to_delete = [
            tid for tid, rec in self._records.items()
            if rec.confirmed_plate is None
            and (current_frame - rec.last_seen_frame) > self.track_timeout
        ]
        for tid in to_delete:
            del self._records[tid]
        return to_delete

    def all_confirmed(self) -> Dict[int, str]:
        """Trả về dict {track_id: confirmed_plate} của tất cả xe đã chốt."""
        return {
            tid: rec.confirmed_plate
            for tid, rec in self._records.items()
            if rec.confirmed_plate is not None
        }

    def active_track_ids(self) -> List[int]:
        """Trả về danh sách track_id đang được theo dõi."""
        return list(self._records.keys())
