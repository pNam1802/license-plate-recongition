"""
draw_zone.py — Vẽ 2 Zone of Interest trên video / webcam
=========================================================
Cách dùng:
    python tools/draw_zone.py --source video.mp4
    python tools/draw_zone.py --source 0

Thao tác:
    Click trái    — thêm điểm vào zone đang active
    Click phải    — xóa điểm vừa thêm
    Tab           — chuyển đổi giữa Zone 1 và Zone 2
    R             — xóa hết điểm của zone đang active
    E             — xóa hết tất cả zones
    Enter / C     — lưu cả 2 zone vào data/zones.json
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

ZONES_FILE = "data/zones.json"
WIN = "Draw Zones  |  Tab=switch zone  Click=diem  R=reset  Enter=luu  Q=thoat"

# Màu cho từng zone (BGR)
ZONE_COLORS = [
    (0, 200, 255),   # Vàng  — Zone 1
    (0, 255, 100),   # Xanh lá — Zone 2
]
ZONE_NAMES = ["Zone 1", "Zone 2"]
NUM_ZONES = 2

zones_points: list[list] = [[], []]
active_zone: int = 0


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        zones_points[active_zone].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if zones_points[active_zone]:
            zones_points[active_zone].pop()


def _draw(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()

    for zi in range(NUM_ZONES):
        pts = zones_points[zi]
        color = ZONE_COLORS[zi]
        is_active = zi == active_zone

        # Màu mờ hơn cho zone không active
        draw_color = color if is_active else tuple(c // 3 for c in color)
        alpha = 0.18 if is_active else 0.08
        thickness = 2 if is_active else 1

        if len(pts) >= 3:
            poly = np.array(pts, dtype=np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [poly], draw_color)
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
            cv2.polylines(out, [poly], isClosed=True, color=draw_color, thickness=thickness)

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(out, pts[i], pts[i + 1], draw_color, thickness)

        for i, p in enumerate(pts):
            cv2.circle(out, p, 7 if is_active else 5, draw_color, -1)
            cv2.circle(out, p, 7 if is_active else 5, (0, 0, 0), 1)
            if is_active:
                cv2.putText(out, str(i + 1), (p[0] + 9, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(out, str(i + 1), (p[0] + 9, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Label zone
        if len(pts) >= 3:
            poly_np = np.array(pts, dtype=np.int32)
            M = cv2.moments(poly_np)
            if M["m00"] != 0:
                lx = int(M["m10"] / M["m00"])
                ly = int(M["m01"] / M["m00"])
            else:
                lx, ly = pts[0]
            tag = f"[{ZONE_NAMES[zi]}]"
            cv2.putText(out, tag, (lx - 35, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 3)
            cv2.putText(out, tag, (lx - 35, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, draw_color, 1)

    # Thanh trạng thái
    h, w = out.shape[:2]
    bar = np.zeros((56, w, 3), dtype=np.uint8)

    z1_status = f"{ZONE_NAMES[0]}: {len(zones_points[0])} pts"
    z2_status = f"{ZONE_NAMES[1]}: {len(zones_points[1])} pts"
    active_label = f"  <<  DANG CHINH SUA: {ZONE_NAMES[active_zone]}  >>"

    dim1 = ZONE_COLORS[0] if active_zone == 0 else tuple(c // 3 for c in ZONE_COLORS[0])
    dim2 = ZONE_COLORS[1] if active_zone == 1 else tuple(c // 3 for c in ZONE_COLORS[1])
    cv2.putText(bar, z1_status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dim1, 1)
    cv2.putText(bar, z2_status, (10 + 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dim2, 1)
    cv2.putText(bar, active_label, (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, ZONE_COLORS[active_zone], 1)

    return np.vstack([out, bar])


def main():
    global active_zone

    parser = argparse.ArgumentParser(description="Vẽ 2 Zone of Interest")
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

    # Load zones cũ nếu có
    if os.path.exists(ZONES_FILE):
        try:
            with open(ZONES_FILE) as f:
                cfg = json.load(f)
            for i, z in enumerate(cfg.get("zones", [])[:NUM_ZONES]):
                if z.get("type") == "polygon":
                    zones_points[i].extend(tuple(p) for p in z["points"])
            counts = [len(zones_points[i]) for i in range(NUM_ZONES)]
            print(f"[INFO] Đã load zones từ {ZONES_FILE}: {counts[0]} + {counts[1]} điểm")
        except Exception:
            pass
    # Fallback: load zone đơn cũ
    elif os.path.exists("data/zone.json"):
        try:
            with open("data/zone.json") as f:
                cfg = json.load(f)
            if cfg.get("type") == "polygon":
                zones_points[0].extend(tuple(p) for p in cfg["points"])
                print(f"[INFO] Đã load zone cũ từ data/zone.json vào Zone 1")
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
    print("  Click trái  — thêm điểm vào zone đang active")
    print("  Click phải  — xóa điểm cuối")
    print("  Tab         — chuyển giữa Zone 1 / Zone 2")
    print("  Space       — frame kế tiếp")
    print("  R           — reset zone đang active")
    print("  E           — reset tất cả zones")
    print("  Enter / C   — lưu và thoát")
    print("  Q           — thoát không lưu\n")

    while True:
        cv2.imshow(WIN, _draw(base_frame))
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("[INFO] Thoát không lưu.")
            break

        elif key == ord('\t'):  # Tab
            active_zone = 1 - active_zone
            print(f"[INFO] Chuyển sang {ZONE_NAMES[active_zone]}")

        elif key in (13, ord('c')):  # Enter hoặc C
            counts = [len(zones_points[i]) for i in range(NUM_ZONES)]
            if any(c < 3 for c in counts):
                missing = [ZONE_NAMES[i] for i, c in enumerate(counts) if c < 3]
                print(f"[WARN] Cần ít nhất 3 điểm cho: {', '.join(missing)}")
                continue

            os.makedirs("data", exist_ok=True)
            zones_cfg = []
            for i in range(NUM_ZONES):
                zones_cfg.append({
                    "name": ZONE_NAMES[i],
                    "type": "polygon",
                    "points": [list(p) for p in zones_points[i]],
                })
            cfg = {"version": 2, "zones": zones_cfg}
            with open(ZONES_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"\n[LƯU] Zones đã lưu: {ZONES_FILE}")
            for i in range(NUM_ZONES):
                print(f"      {ZONE_NAMES[i]}: {len(zones_points[i])} điểm — {zones_points[i]}")
            print()
            break

        elif key == ord('r'):
            zones_points[active_zone].clear()
            print(f"[INFO] Đã reset {ZONE_NAMES[active_zone]}.")

        elif key == ord('e'):
            for i in range(NUM_ZONES):
                zones_points[i].clear()
            print("[INFO] Đã reset tất cả zones.")

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
