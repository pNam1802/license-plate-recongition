"""
draw_zone.py — Vẽ Zone of Interest trên video / webcam
=======================================================
Cách dùng:
    python tools/draw_zone.py --source video.mp4
    python tools/draw_zone.py --source 0

Thao tác:
    Click trái    — thêm điểm vào polygon
    Click phải    — xóa điểm vừa thêm
    Enter / C     — lưu zone vào data/zone.json
    R             — xóa hết, bắt đầu lại
    Space         — chuyển sang frame kế tiếp
    Q             — thoát không lưu
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ZONE_FILE = "data/zone.json"
WIN       = "Draw Zone  |  Click trai=them diem  Click phai=xoa  Enter=luu  R=reset  Q=thoat"
COLOR     = (0, 200, 255)

_points: list = []


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _points:
            _points.pop()


def _draw(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    pts = _points

    if len(pts) >= 3:
        poly = np.array(pts, dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], COLOR)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
        cv2.polylines(out, [poly], isClosed=True, color=COLOR, thickness=2)

    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i + 1], COLOR, 2)

    for i, p in enumerate(pts):
        cv2.circle(out, p, 7, COLOR, -1)
        cv2.circle(out, p, 7, (0, 0, 0), 1)
        cv2.putText(out, str(i + 1), (p[0] + 9, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(out, str(i + 1), (p[0] + 9, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    h, w = out.shape[:2]
    bar = np.zeros((48, w, 3), dtype=np.uint8)
    status = f"Diem: {len(pts)}"
    if len(pts) < 3:
        status += "  (can it nhat 3 diem)"
    else:
        status += "  — nhan Enter de luu"
    cv2.putText(bar, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 1)
    return np.vstack([out, bar])


def main():
    parser = argparse.ArgumentParser(description="Vẽ Zone of Interest")
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

    if os.path.exists(ZONE_FILE):
        try:
            with open(ZONE_FILE) as f:
                cfg = json.load(f)
            if cfg.get("type") == "polygon":
                _points.extend(tuple(p) for p in cfg["points"])
                print(f"[INFO] Đã load zone cũ từ {ZONE_FILE} ({len(_points)} điểm)")
        except Exception:
            pass

    ret, base_frame = cap.read()
    if not ret:
        print("[ERR] Không đọc được frame từ nguồn")
        sys.exit(1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(1280, base_frame.shape[1]), min(760, base_frame.shape[0] + 48))
    cv2.setMouseCallback(WIN, _mouse_cb)

    print("\nThao tác:")
    print("  Click trái  — thêm điểm")
    print("  Click phải  — xóa điểm cuối")
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
            if len(_points) < 3:
                print("[WARN] Cần ít nhất 3 điểm để tạo polygon zone.")
                continue
            os.makedirs("data", exist_ok=True)
            cfg = {"type": "polygon", "points": [list(p) for p in _points]}
            with open(ZONE_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"\n[LƯU] Zone đã lưu: {ZONE_FILE}")
            print(f"      {len(_points)} điểm: {_points}\n")
            break

        elif key == ord('r'):
            _points.clear()
            print("[INFO] Reset — đã xóa hết điểm.")

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
