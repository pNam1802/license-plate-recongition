"""
count_vehicles.py — Đếm xe theo đường kẻ ảo, phân loại từng loại
=================================================================
Cách dùng:
    python tools/count_vehicles.py --source video.mp4
    python tools/count_vehicles.py --source 0              # webcam
    python tools/count_vehicles.py --source video.mp4 --skip 2

Yêu cầu:
    Vẽ đường kẻ trước bằng: python tools/draw_line.py --source video.mp4
    Kết quả lưu tại data/line.json (tự động load)

    Nếu không có line.json → đếm tất cả xe xuất hiện trong frame (không theo đường kẻ).

Phím tắt:
    Q / ESC — Thoát
    P       — Tạm dừng / Tiếp tục
    R       — Reset bộ đếm
    S       — Lưu screenshot
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.vehicle_detector import VehicleDetector
from pipeline.zone_filter import LineFilter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VEHICLE_CONF  = 0.35
MIN_AREA      = 4000
SKIP_FRAMES   = 2
TRACK_TIMEOUT = 60    # frame không thấy thì xóa track khỏi bộ nhớ

LINE_FILE     = "data/line.json"
SCREENSHOT_DIR = "data/screenshots"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Màu theo loại xe (BGR)
CLASS_COLORS: Dict[str, tuple] = {
    "car":        (50,  200,  50),   # Xanh lá
    "motorcycle": (50,  180, 255),   # Xanh dương nhạt
    "truck":      (50,   50, 255),   # Đỏ
    "bus":        (180,  50, 255),   # Tím
    "vehicle":    (120, 200, 120),   # Xanh nhạt (fallback)
}

# Nhãn tiếng Việt
CLASS_LABELS: Dict[str, str] = {
    "car":        "Oto",
    "motorcycle": "Xe may",
    "truck":      "Xe tai",
    "bus":        "Xe bus",
    "vehicle":    "Xe",
}


def _get_color(class_name: str) -> tuple:
    name = class_name.lower()
    for key, color in CLASS_COLORS.items():
        if key in name:
            return color
    return CLASS_COLORS["vehicle"]


def _get_label(class_name: str) -> str:
    name = class_name.lower()
    for key, label in CLASS_LABELS.items():
        if key in name:
            return label
    return class_name


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_counter_panel(
    frame: np.ndarray,
    counts: Dict[str, int],
    total: int,
    fps: float,
    paused: bool,
    has_line: bool,
) -> None:
    """Vẽ bảng đếm góc trên-trái."""
    h, w = frame.shape[:2]
    PANEL_W  = 230
    LINE_H   = 34
    PAD      = 12
    n_rows   = max(1, len(counts)) + 2   # +2 cho header và total
    panel_h  = PAD + 28 + 6 + LINE_H * n_rows + PAD

    # Nền trong suốt
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + PANEL_W, 10 + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + PANEL_W, 10 + panel_h), (60, 200, 60), 1)

    x = 22
    y = 10 + PAD + 18

    # Tiêu đề
    title = "VE COUNT" if has_line else "VE COUNT (no line)"
    cv2.putText(frame, title, (x, y), FONT, 0.65, (60, 200, 60), 2)
    y += 8
    cv2.line(frame, (x, y), (10 + PANEL_W - PAD, y), (60, 200, 60), 1)
    y += LINE_H - 6

    # FPS / Pause
    if paused:
        cv2.putText(frame, "[ PAUSED ]", (x, y), FONT, 0.50, (80, 120, 255), 1)
    else:
        cv2.putText(frame, f"FPS: {fps:.1f}", (x, y), FONT, 0.50, (140, 200, 140), 1)
    y += LINE_H - 4

    # Từng loại xe
    for cls_name, cnt in sorted(counts.items()):
        color = _get_color(cls_name)
        label = _get_label(cls_name)
        cv2.putText(frame, f"{label}:", (x, y), FONT, 0.58, color, 1)
        cv2.putText(frame, str(cnt), (x + PANEL_W - 46, y), FONT, 0.70, color, 2)
        y += LINE_H

    # Tổng
    cv2.line(frame, (x, y - 6), (10 + PANEL_W - PAD, y - 6), (100, 100, 100), 1)
    cv2.putText(frame, "Tong:", (x, y + 4), FONT, 0.60, (255, 255, 255), 1)
    cv2.putText(frame, str(total), (x + PANEL_W - 46, y + 4), FONT, 0.80, (255, 255, 100), 2)


def _draw_box(frame: np.ndarray, box: tuple, label: str, color: tuple) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 1)
    bg_y = max(0, y1 - th - 6)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, bg_y), (x1 + tw + 6, y1), color, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), FONT, 0.50, (0, 0, 0), 2)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), FONT, 0.50, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class _FPS:
    def __init__(self, n=30):
        self._t = []
        self._n = n

    def tick(self):
        self._t.append(time.monotonic())
        if len(self._t) > self._n:
            self._t.pop(0)

    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


# ---------------------------------------------------------------------------
# Track state — dọn dẹp track cũ
# ---------------------------------------------------------------------------

class _TrackMemory:
    def __init__(self, timeout: int):
        self._timeout  = timeout
        self._last_seen: Dict[int, int] = {}

    def update(self, track_id: int, frame_idx: int):
        self._last_seen[track_id] = frame_idx

    def cleanup(self, frame_idx: int):
        dead = [tid for tid, f in self._last_seen.items()
                if frame_idx - f > self._timeout]
        for tid in dead:
            del self._last_seen[tid]
        return dead


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(source, skip: int, conf: float, min_area: int, device: str):
    # ---- Load models ----
    print("\n" + "=" * 50)
    print("  VEHICLE COUNTER — Đang khởi tạo...")
    print("=" * 50)

    detector = VehicleDetector(conf=conf, min_area=min_area, device=device)

    # ---- Load line ----
    line_filter = None
    if os.path.exists(LINE_FILE):
        with open(LINE_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        line_filter = LineFilter.from_config(cfg)
        side = cfg.get("entry_side", 0)
        side_str = {1: "+1 (phải/dưới)", -1: "-1 (trái/trên)", 0: "cả 2 chiều"}.get(side, str(side))
        print(f"[INFO] Line: P1={line_filter.p1}  P2={line_filter.p2}  chiều vào={side_str}")
    else:
        print(f"[WARN] Không tìm thấy {LINE_FILE} — đếm tất cả xe xuất hiện trong frame")
        print(f"[WARN] Chạy 'python tools/draw_line.py' để vẽ đường kẻ")

    # ---- Mở video ----
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được nguồn: {source}")
        sys.exit(1)

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    try:
        import tkinter as tk
        root = tk.Tk()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        sw, sh = 1920, 1080

    scale = min(sw * 0.85 / vw, sh * 0.85 / vh)
    WIN = "Vehicle Counter  |  Q=thoat  P=pause  R=reset"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, int(vw * scale), int(vh * scale))

    # ---- Trạng thái ----
    fps_counter  = _FPS()
    mem          = _TrackMemory(TRACK_TIMEOUT)
    frame_idx    = 0
    paused       = False
    last_frame   = None
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Bộ đếm
    crossed_ids: Set[int]       = set()   # ID đã vượt đường (tránh đếm lại)
    counts: Dict[str, int]      = defaultdict(int)   # {class_name: count}
    class_of: Dict[int, str]    = {}      # {track_id: class_name}

    # Nếu không có line → đếm mỗi track_id mới xuất hiện lần đầu
    seen_ids: Set[int] = set()

    print(f"\n[INFO] Nguồn: {source}")
    print(f"[INFO] Phím: Q=thoát  P=pause  R=reset  S=screenshot\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Hết video.")
                break
            frame_idx += 1

            if frame_idx % skip != 0:
                if last_frame is not None:
                    cv2.imshow(WIN, last_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('p'):
                    paused = not paused
                continue

            vehicles = detector.track(frame)
            annotated = frame.copy()

            # Dọn dẹp track cũ
            dead = mem.cleanup(frame_idx)
            for tid in dead:
                class_of.pop(tid, None)
                if line_filter:
                    line_filter.remove_track(tid)

            # Vẽ đường kẻ
            if line_filter:
                line_filter.draw(annotated)

            for v in vehicles:
                mem.update(v.track_id, frame_idx)
                color = _get_color(v.class_name)
                label = f"{_get_label(v.class_name)} #{v.track_id}"

                if line_filter:
                    # Đếm khi vượt đường kẻ
                    crossed = line_filter.check_crossing(v.track_id, v.box)
                    if crossed and v.track_id not in crossed_ids:
                        crossed_ids.add(v.track_id)
                        class_of[v.track_id] = v.class_name
                        counts[v.class_name] += 1
                        total = sum(counts.values())
                        print(f"  [COUNT] #{v.track_id} {_get_label(v.class_name)}"
                              f"  — tổng: {total}")
                    # Highlight xe vừa vượt
                    if v.track_id in crossed_ids:
                        color = tuple(min(255, c + 80) for c in color)
                else:
                    # Không có line → đếm lần đầu xuất hiện
                    if v.track_id not in seen_ids:
                        seen_ids.add(v.track_id)
                        counts[v.class_name] += 1
                        total = sum(counts.values())
                        print(f"  [COUNT] #{v.track_id} {_get_label(v.class_name)}"
                              f"  — tổng: {total}")

                _draw_box(annotated, v.box, label, color)

            fps_counter.tick()
            total = sum(counts.values())
            _draw_counter_panel(annotated, dict(counts), total,
                                fps_counter.fps, paused, line_filter is not None)

            last_frame = annotated
            cv2.imshow(WIN, annotated)

        else:
            if last_frame is not None:
                disp = last_frame.copy()
                h, w = disp.shape[:2]
                cv2.putText(disp, "[ PAUSED ]", (w // 2 - 90, h // 2),
                            FONT, 1.2, (80, 120, 255), 3)
                cv2.imshow(WIN, disp)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Tạm dừng' if paused else 'Tiếp tục'}")
        elif key == ord('r'):
            crossed_ids.clear()
            seen_ids.clear()
            counts.clear()
            class_of.clear()
            detector.reset()
            frame_idx = 0
            print("[INFO] Reset bộ đếm.")
        elif key == ord('s') and last_frame is not None:
            path = os.path.join(SCREENSHOT_DIR, f"count_{int(time.time())}.jpg")
            cv2.imwrite(path, last_frame)
            print(f"[INFO] Screenshot: {path}")

    cap.release()
    cv2.destroyAllWindows()

    # ---- Kết quả cuối ----
    print("\n" + "=" * 50)
    print("  KẾT QUẢ ĐẾM XE")
    print("=" * 50)
    if counts:
        for cls_name, cnt in sorted(counts.items()):
            print(f"  {_get_label(cls_name):<12} : {cnt}")
        print(f"  {'TỔNG':<12} : {sum(counts.values())}")
    else:
        print("  Không đếm được xe nào.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Đếm xe theo đường kẻ ảo, phân loại từng loại",
    )
    parser.add_argument("--source", "-s", default="0",
                        help="Webcam (0), file video, hoặc RTSP URL")
    parser.add_argument("--skip", "-k", type=int, default=SKIP_FRAMES,
                        help=f"Xử lý 1 trong mỗi N frames (mặc định: {SKIP_FRAMES})")
    parser.add_argument("--conf", type=float, default=VEHICLE_CONF,
                        help=f"Ngưỡng confidence detect xe (mặc định: {VEHICLE_CONF})")
    parser.add_argument("--min-area", type=int, default=MIN_AREA,
                        help=f"Diện tích xe tối thiểu px² (mặc định: {MIN_AREA})")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="Thiết bị tính toán (mặc định: cpu)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(
        source   = source,
        skip     = args.skip,
        conf     = args.conf,
        min_area = args.min_area,
        device   = args.device,
    )


if __name__ == "__main__":
    main()
