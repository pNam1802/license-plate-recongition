"""
draw_line.py — Vẽ đường kẻ ảo (virtual tripwire) để đếm xe
============================================================
Cách dùng:
    python tools/draw_line.py --source video.mp4
    python tools/draw_line.py --source 0

Thao tác:
    Click trái    — đặt điểm (tối đa 2 điểm = 1 đường)
    Click phải    — xóa điểm vừa đặt
    F             — đảo chiều vào (flip entry direction)
    R             — reset
    Space         — chuyển sang frame kế tiếp
    Enter / C     — lưu vào data/line.json
    Q             — thoát không lưu
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LINE_FILE = "data/line.json"
WIN       = "Draw Line  |  Click=dat diem  F=doi chieu  Enter=luu  Q=thoat"
COLOR     = (0, 200, 255)
COLOR_ENTRY = (0, 255, 80)

_points: list = []
_entry_side: int = 1   # +1 hoặc -1; 0 = đếm cả 2 chiều


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_points) < 2:
            _points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _points:
            _points.pop()


def _draw_arrow(img: np.ndarray, p1, p2, entry_side: int) -> None:
    """Vẽ mũi tên chỉ chiều vào theo vector pháp tuyến của đường."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy) or 1.0
    nx = -dy / length   # pháp tuyến → trỏ về phía +1
    ny =  dx / length

    mx = (p1[0] + p2[0]) // 2
    my = (p1[1] + p2[1]) // 2

    off = 55
    ax1 = int(mx + nx * entry_side * off)
    ay1 = int(my + ny * entry_side * off)
    ax2 = int(mx - nx * entry_side * 15)
    ay2 = int(my - ny * entry_side * 15)
    cv2.arrowedLine(img, (ax1, ay1), (ax2, ay2), COLOR_ENTRY, 3, tipLength=0.35)

    lx = int(mx + nx * entry_side * (off + 18))
    ly = int(my + ny * entry_side * (off + 18))
    cv2.putText(img, "VAO", (lx - 18, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ENTRY, 2)


def _draw(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()

    if len(_points) == 2:
        cv2.line(out, _points[0], _points[1], COLOR, 3)
        _draw_arrow(out, _points[0], _points[1], _entry_side)

    for i, p in enumerate(_points):
        cv2.circle(out, p, 8, COLOR, -1)
        cv2.circle(out, p, 8, (0, 0, 0), 1)
        label = f"P{i + 1}  ({p[0]}, {p[1]})"
        cv2.putText(out, label, (p[0] + 12, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, label, (p[0] + 12, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    h, w = out.shape[:2]
    bar = np.zeros((56, w, 3), dtype=np.uint8)

    if len(_points) == 0:
        status = "Click de dat P1"
        dir_info = ""
    elif len(_points) == 1:
        status = f"P1=({_points[0][0]},{_points[0][1]})  —  Click de dat P2"
        dir_info = ""
    else:
        dx = _points[1][0] - _points[0][0]
        dy = _points[1][1] - _points[0][1]
        length = int((dx**2 + dy**2) ** 0.5)
        status = (f"P1=({_points[0][0]},{_points[0][1]})  "
                  f"P2=({_points[1][0]},{_points[1][1]})  "
                  f"len={length}px")
        side_label = "+1 (phai/duoi)" if _entry_side == 1 else "-1 (trai/tren)"
        dir_info = f"  |  Chieu vao: {side_label}  [F=doi chieu]  —  Enter=luu"

    cv2.putText(bar, status + dir_info, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR, 1)
    cv2.putText(bar, "F=doi chieu vao  |  R=reset  |  Space=frame ke  |  Enter/C=luu  |  Q=thoat",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    return np.vstack([out, bar])


def main():
    global _entry_side

    parser = argparse.ArgumentParser(description="Vẽ đường kẻ ảo đếm xe")
    parser.add_argument("--source", "-s", default="0",
                        help="Nguồn video: số thiết bị (0), file .mp4, hoặc RTSP URL")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERR] Không mở được nguồn: {source}")
        sys.exit(1)

    # Load line cũ nếu có
    if os.path.exists(LINE_FILE):
        try:
            with open(LINE_FILE) as f:
                cfg = json.load(f)
            if cfg.get("type") == "line":
                _points.extend(tuple(p) for p in [cfg["p1"], cfg["p2"]])
                _entry_side = cfg.get("entry_side", 1)
                print(f"[INFO] Đã load line cũ: P1={_points[0]}  P2={_points[1]}  entry_side={_entry_side}")
        except Exception:
            pass

    ret, base_frame = cap.read()
    if not ret:
        print("[ERR] Không đọc được frame từ nguồn")
        sys.exit(1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(1280, base_frame.shape[1]), min(760, base_frame.shape[0] + 56))
    cv2.setMouseCallback(WIN, _mouse_cb)

    print("\nThao tác:")
    print("  Click trái  — đặt điểm (tối đa 2)")
    print("  Click phải  — xóa điểm cuối")
    print("  F           — đảo chiều vào (flip entry_side +1/-1)")
    print("  Space       — frame kế tiếp")
    print("  R           — reset")
    print("  Enter / C   — lưu và thoát")
    print("  Q           — thoát không lưu\n")

    while True:
        cv2.imshow(WIN, _draw(base_frame))
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("[INFO] Thoát không lưu.")
            break

        elif key in (13, ord('c')):
            if len(_points) < 2:
                print("[WARN] Cần đặt đủ 2 điểm.")
                continue
            os.makedirs("data", exist_ok=True)
            cfg = {
                "type":       "line",
                "p1":         list(_points[0]),
                "p2":         list(_points[1]),
                "entry_side": _entry_side,
            }
            with open(LINE_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"\n[LƯU] Line đã lưu: {LINE_FILE}")
            print(f"      P1={_points[0]}  P2={_points[1]}  entry_side={_entry_side}\n")
            break

        elif key == ord('f'):
            _entry_side = -_entry_side
            side_label = "+1 (phai/duoi)" if _entry_side == 1 else "-1 (trai/tren)"
            print(f"[INFO] Đổi chiều vào → entry_side={_entry_side} ({side_label})")

        elif key == ord('r'):
            _points.clear()
            _entry_side = 1
            print("[INFO] Reset.")

        elif key == ord(' '):
            ret, frame = cap.read()
            if ret:
                base_frame = frame
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret:
                    base_frame = frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
