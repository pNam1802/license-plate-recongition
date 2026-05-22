"""
Zone of Interest (ZoI) Filter
==============================
Xác định vùng trên màn hình nơi biển số xe thường hiện rõ nhất.
Chỉ kích hoạt OCR khi xe đi vào vùng này.

Hai chế độ:
  1. Horizontal Band  — Dải ngang theo % chiều cao frame (mặc định)
                        Phù hợp camera cố định nhìn thẳng vào đường
  2. Polygon          — Đa giác tùy ý định nghĩa bằng tọa độ pixel
                        Phù hợp hơn khi góc camera nghiêng

Cách dùng:
    # Dùng dải ngang: ZoI là phần giữa frame, từ 25% đến 80% chiều cao
    zf = ZoneFilter(top_ratio=0.25, bottom_ratio=0.80)

    # Dùng polygon
    zf = ZoneFilter(polygon=[(100,200), (800,200), (900,500), (50,500)])

    # Dùng nhiều zones (từ draw_zone.py mới)
    mzf = MultiZoneFilter.from_config({"version": 2, "zones": [...]})

    # Kiểm tra xe có trong zone không
    if zf.is_in_zone(vehicle_box, frame.shape):
        ...

    # Vẽ zone lên frame để visualize
    frame = zf.draw(frame)
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ZoneFilter:
    """
    Bộ lọc Zone of Interest cho pipeline nhận diện biển số.

    Kiểm tra tâm của vehicle bounding box có nằm trong vùng định nghĩa không.
    """

    def __init__(
        self,
        polygon:      Optional[List[Tuple[int, int]]] = None,
        top_ratio:    float = 0.20,   # % chiều cao frame — cạnh trên của dải ngang
        bottom_ratio: float = 0.85,   # % chiều cao frame — cạnh dưới của dải ngang
    ):
        """
        Args:
            polygon:      Danh sách tọa độ pixel [(x,y)...] định nghĩa vùng.
                          Nếu None → dùng horizontal band.
            top_ratio:    Cạnh trên của dải ngang (0.0 → 1.0 theo chiều cao frame).
            bottom_ratio: Cạnh dưới của dải ngang.
        """
        self.polygon      = polygon
        self.top_ratio    = top_ratio
        self.bottom_ratio = bottom_ratio

        # Cache polygon numpy array
        self._poly_np: Optional[np.ndarray] = None
        if polygon:
            self._poly_np = np.array(polygon, dtype=np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_in_zone(
        self,
        vehicle_box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        """
        Kiểm tra TẤT CẢ 4 góc của bbox có nằm trong Zone of Interest không.
        Xe chỉ được coi là trong zone khi toàn bộ bounding box nằm bên trong.

        Args:
            vehicle_box: (x1, y1, x2, y2) bounding box của xe
            frame_shape: (height, width, channels) của frame

        Returns:
            True nếu cả 4 góc đều nằm trong vùng ZoI
        """
        x1, y1, x2, y2 = vehicle_box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        if self._poly_np is not None:
            return all(
                cv2.pointPolygonTest(self._poly_np, (float(cx), float(cy)), False) >= 0
                for cx, cy in corners
            )
        else:
            fh = frame_shape[0]
            top    = int(fh * self.top_ratio)
            bottom = int(fh * self.bottom_ratio)
            return y1 >= top and y2 <= bottom

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 200, 255),
        alpha: float = 0.15,
        label: bool  = True,
        label_text: str = "ZONE",
    ) -> np.ndarray:
        """
        Vẽ Zone of Interest lên frame (overlay bán trong suốt).

        Args:
            frame: BGR frame
            color: Màu vẽ (BGR)
            alpha: Độ trong suốt của vùng tô (0.0 → 1.0)
            label: Có vẽ nhãn "ZONE" không

        Returns:
            Frame đã được vẽ zone lên
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        if self._poly_np is not None:
            # Vẽ polygon
            cv2.fillPoly(overlay, [self._poly_np], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.polylines(frame, [self._poly_np], isClosed=True, color=color, thickness=2)
            if label:
                # Tính centroid để đặt nhãn
                M = cv2.moments(self._poly_np)
                if M["m00"] != 0:
                    lx = int(M["m10"] / M["m00"])
                    ly = int(M["m01"] / M["m00"])
                else:
                    lx, ly = self._poly_np[0]
                cv2.putText(frame, label_text, (lx - 25, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Không vẽ gì nếu zone phủ toàn frame (đã tắt zone)
            if self.top_ratio == 0.0 and self.bottom_ratio == 1.0:
                return frame
            top    = int(h * self.top_ratio)
            bottom = int(h * self.bottom_ratio)
            cv2.rectangle(overlay, (0, top), (w, bottom), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.line(frame, (0, top),    (w, top),    color, 2)
            cv2.line(frame, (0, bottom), (w, bottom), color, 2)
            if label:
                cv2.putText(frame, label_text, (10, top + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    @classmethod
    def from_config(cls, config: dict) -> "ZoneFilter":
        """
        Tạo ZoneFilter từ dict config (tiện cho lưu/load file JSON/YAML).

        Config format:
            {"type": "band",    "top": 0.25, "bottom": 0.80}
            {"type": "polygon", "points": [[x1,y1], [x2,y2], ...]}
        """
        zone_type = config.get("type", "band")
        if zone_type == "polygon":
            pts = [tuple(p) for p in config["points"]]
            return cls(polygon=pts)
        else:
            return cls(
                top_ratio    = config.get("top",    0.25),
                bottom_ratio = config.get("bottom", 0.80),
            )


class MultiZoneFilter:
    """
    Bộ lọc nhiều Zone of Interest — xe trong bất kỳ zone nào đều được xử lý.

    Dùng khi camera phủ nhiều làn / nhiều vùng độc lập.
    """

    # Màu BGR cho từng zone
    COLORS: List[Tuple[int, int, int]] = [
        (0, 200, 255),   # Vàng     — Zone 1
        (0, 255, 100),   # Xanh lá  — Zone 2
        (255, 100, 0),   # Cam      — Zone 3
        (200, 0, 255),   # Tím      — Zone 4
    ]

    def __init__(
        self,
        zones: List[ZoneFilter],
        names: Optional[List[str]] = None,
    ):
        self.zones = zones
        self.names = names or [f"Zone {i + 1}" for i in range(len(zones))]

    def is_in_zone(
        self,
        vehicle_box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        """Trả về True nếu xe nằm trong BẤT KỲ zone nào."""
        return any(z.is_in_zone(vehicle_box, frame_shape) for z in self.zones)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ tất cả zones lên frame, mỗi zone một màu khác nhau."""
        for i, (zone, name) in enumerate(zip(self.zones, self.names)):
            color = self.COLORS[i % len(self.COLORS)]
            zone.draw(frame, color=color, label=True, label_text=name)
        return frame

    @classmethod
    def from_config(cls, config: dict) -> "MultiZoneFilter":
        """
        Tạo MultiZoneFilter từ dict config của data/zones.json.

        Config format (version 2):
            {
              "version": 2,
              "zones": [
                {"name": "Zone 1", "type": "polygon", "points": [...]},
                {"name": "Zone 2", "type": "polygon", "points": [...]}
              ]
            }
        """
        zones_cfg = config.get("zones", [])
        zones = []
        names = []
        for z in zones_cfg:
            zones.append(ZoneFilter.from_config(z))
            names.append(z.get("name", f"Zone {len(zones)}"))
        return cls(zones, names)


class LineFilter:
    """
    Đường kẻ ảo (virtual tripwire) để đếm xe đi qua theo một chiều.

    entry_side: chiều xe được tính vào count.
      +1  → xe xuất phát từ phía +1 (phải/dưới tùy góc) vượt sang phía -1
      -1  → xe xuất phát từ phía -1 (trái/trên tùy góc) vượt sang phía +1
       0  → đếm cả 2 chiều (mặc định cũ)

    data/line.json format:
        {"type": "line", "p1": [x1,y1], "p2": [x2,y2], "entry_side": 1}
    """

    COLOR       = (0, 200, 255)
    COLOR_ENTRY = (0, 255, 80)    # Xanh lá — mũi tên chiều vào

    def __init__(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        entry_side: int = 0,
    ):
        self.p1          = p1
        self.p2          = p2
        self.entry_side  = entry_side   # +1 / -1 / 0
        self._prev_side: Dict[int, int] = {}

    # ------------------------------------------------------------------

    def _side(self, cx: int, cy: int) -> int:
        """Dấu cross product: +1 (phải/dưới P1→P2) / -1 (trái/trên) / 0."""
        val = (self.p2[0] - self.p1[0]) * (cy - self.p1[1]) \
            - (self.p2[1] - self.p1[1]) * (cx - self.p1[0])
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0

    def check_crossing(
        self,
        track_id: int,
        box: Tuple[int, int, int, int],
    ) -> bool:
        """
        True nếu xe vừa vượt qua đường theo đúng chiều entry_side.
        entry_side=0 → chấp nhận cả 2 chiều.
        """
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        side = self._side(cx, cy)

        prev = self._prev_side.get(track_id)
        self._prev_side[track_id] = side

        if prev is None or side == 0 or prev == 0:
            return False
        if prev == side:
            return False

        # Crossed — kiểm tra chiều
        if self.entry_side == 0:
            return True
        # entry_side=+1: xe xuất phát từ +1 sang -1
        return prev == self.entry_side

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ đường kẻ + mũi tên chỉ chiều vào."""
        import math

        cv2.line(frame, self.p1, self.p2, self.COLOR, 3)
        cv2.circle(frame, self.p1, 6, self.COLOR, -1)
        cv2.circle(frame, self.p2, 6, self.COLOR, -1)

        # Vector pháp tuyến (pointing to +1 side)
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        nx = -dy / length
        ny =  dx / length

        mx = (self.p1[0] + self.p2[0]) // 2
        my = (self.p1[1] + self.p2[1]) // 2

        if self.entry_side != 0:
            # Mũi tên: từ phía entry_side → vượt qua đường
            off = 55
            ax1 = int(mx + nx * self.entry_side * off)
            ay1 = int(my + ny * self.entry_side * off)
            ax2 = int(mx - nx * self.entry_side * 15)
            ay2 = int(my - ny * self.entry_side * 15)
            cv2.arrowedLine(frame, (ax1, ay1), (ax2, ay2),
                            self.COLOR_ENTRY, 3, tipLength=0.35)
            # Chữ "VÀO" bên cạnh mũi tên
            lx = int(mx + nx * self.entry_side * (off + 18))
            ly = int(my + ny * self.entry_side * (off + 18))
            cv2.putText(frame, "VAO", (lx - 18, ly + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_ENTRY, 2)
        else:
            cv2.putText(frame, "LINE", (mx - 20, my - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR, 2)

        return frame

    def remove_track(self, track_id: int) -> None:
        self._prev_side.pop(track_id, None)

    @classmethod
    def from_config(cls, config: dict) -> "LineFilter":
        return cls(
            p1         = tuple(config["p1"]),
            p2         = tuple(config["p2"]),
            entry_side = config.get("entry_side", 0),
        )
