"""
Stage 1 + Stage 2: Vehicle Detection & ByteTrack Tracking
==========================================================
- Load yolo26n.pt để phát hiện ô tô, xe máy, xe tải, xe buýt
- Dùng .track() với ByteTrack để gán ID duy nhất cho mỗi xe
- Lọc bỏ detection có confidence thấp hoặc bounding box quá nhỏ
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VEHICLE_MODEL_PATH = "models/yolo26n.pt"

# YOLO COCO class IDs cho phương tiện
VEHICLE_CLASS_IDS = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

DEFAULT_CONF       = 0.40   # Ngưỡng confidence tối thiểu
DEFAULT_MIN_AREA   = 4000   # Diện tích bbox tối thiểu (px²) — loại xe ở quá xa
DEFAULT_IOU        = 0.45  # IOU threshold cho NMS


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class VehicleDetection:
    track_id:   int           # ByteTrack ID duy nhất
    class_name: str           # "car" / "motorcycle" / "bus" / "truck"
    box:        Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    conf:       float


# ---------------------------------------------------------------------------
# VehicleDetector
# ---------------------------------------------------------------------------

class VehicleDetector:
    """
    Bọc YOLOv8 + ByteTrack để phát hiện và theo dõi phương tiện.

    Ví dụ:
        detector = VehicleDetector()
        vehicles = detector.track(frame)
        for v in vehicles:
            print(v.track_id, v.class_name, v.box)
    """

    def __init__(
        self,
        model_path: str = VEHICLE_MODEL_PATH,
        conf: float     = DEFAULT_CONF,
        min_area: int   = DEFAULT_MIN_AREA,
        iou: float      = DEFAULT_IOU,
        device: str     = "cpu",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[VehicleDetector] Không tìm thấy model: {model_path}"
            )
        self.model    = YOLO(model_path)
        self.conf     = conf
        self.min_area = min_area
        self.iou      = iou
        self.device   = device

        # Kiểm tra xem model có class phương tiện không
        model_names = self.model.names  # dict {id: name}
        self._valid_ids = {
            k for k, v in model_names.items()
            if v.lower() in {c.lower() for c in VEHICLE_CLASS_IDS.values()}
        }
        # Fallback về COCO IDs nếu model dùng tên chuẩn
        if not self._valid_ids:
            self._valid_ids = set(VEHICLE_CLASS_IDS.keys())

        print(f"[VehicleDetector] Model: {model_path}")
        print(f"[VehicleDetector] Tracking classes: {[model_names.get(i, i) for i in self._valid_ids]}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Chạy detection + tracking trên một frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            Danh sách VehicleDetection đã lọc và có ID.
        """
        results = self.model.track(
            frame,
            tracker  = "bytetrack.yaml",
            persist  = True,          # Bắt buộc để duy trì ID qua frames
            conf     = self.conf,
            iou      = self.iou,
            classes  = list(self._valid_ids),
            device   = self.device,
            verbose  = False,
        )

        detections: List[VehicleDetection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes_result = results[0].boxes
        model_names  = self.model.names

        for i, box in enumerate(boxes_result):
            # Bỏ qua nếu chưa có tracking ID (frame đầu tiên)
            if box.id is None:
                continue

            track_id   = int(box.id[0])
            class_id   = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Lọc diện tích quá nhỏ
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area:
                continue

            class_name = model_names.get(class_id, "vehicle")

            detections.append(VehicleDetection(
                track_id   = track_id,
                class_name = class_name,
                box        = (x1, y1, x2, y2),
                conf       = conf_score,
            ))

        return detections

    def reset(self):
        """Reset trạng thái tracker (dùng khi chuyển video source)."""
        self.model.predictor = None
