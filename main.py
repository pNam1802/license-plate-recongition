"""
main.py — Entry point: Pipeline Nhận Diện Biển Số Xe Thời Gian Thực
=====================================================================
Kết hợp toàn bộ 7 bước:
  1. Video Ingestion & Frame Skipping
  2. Vehicle Detection (yolo26n.pt)
  3. ByteTrack Tracking
  4. Plate Detection (plate-detect.pt)
  5. Perspective Transform
  6. PaddleOCR
  7. Regex Validation + Voting

Cách dùng:
    python main.py --source 0                          # Webcam
    python main.py --source video.mp4                  # File video
    python main.py --source rtsp://user:pass@ip/stream # RTSP

Phím tắt:
    q  — Thoát
    p  — Tạm dừng / Tiếp tục
    f  — Fullscreen
    s  — Lưu frame hiện tại vào data/debug/
    r  — Reset tracker (xóa tất cả ID hiện tại)
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np

from pipeline.vehicle_detector import VehicleDetector
from pipeline.plate_reader import PlateReader
from pipeline.vote_logic import VoteManager, VehicleState, normalize_plate, is_valid_vn_plate
from pipeline.zone_filter import ZoneFilter


# ---------------------------------------------------------------------------
# Config mặc định
# ---------------------------------------------------------------------------

SKIP_FRAMES   = 2        # Xử lý 1 frame trong mỗi N frames
MIN_AREA      = 4000     # Diện tích xe tối thiểu (px²)
VEHICLE_CONF  = 0.40     # Confidence ngưỡng vehicle detection
TRACK_TIMEOUT = 90       # Frames không thấy thì xóa track
MIN_VOTES     = 2        # Số lần đọc tối thiểu để chốt biển số
VOTE_RATIO    = 0.50     # Tỉ lệ đồng thuận để chốt
OCR_INTERVAL  = 1        # Số frame giữa 2 lần OCR cho cùng 1 xe
MAX_SAMPLES   = 40       # Số lần OCR tối đa trước khi bỏ qua xe
RESULTS_DIR   = "data/results"

# Zone of Interest — dải ngang nơi biển số rõ nhất
# Đặt 0.0 / 1.0 để tắt hoàn toàn (toàn frame đều hợp lệ)
ZONE_TOP      = 0.0      # % chiều cao frame (cạnh trên)
ZONE_BOTTOM   = 1.0      # % chiều cao frame (cạnh dưới)

# Màu sắc (BGR)
COLOR_VEHICLE     = (0,   200, 50)    # Xanh lá — bounding box xe
COLOR_PLATE       = (0,   220, 255)   # Vàng — bounding box biển số
COLOR_CONFIRMED   = (50,  255, 50)    # Xanh sáng — xe đã có biển số chốt
COLOR_TEXT_BG     = (20,  20,  20)    # Nền text tối
COLOR_PANEL_BG    = (15,  15,  15)    # Nền sidebar
COLOR_ACCENT      = (80,  160, 255)   # Xanh dương nhạt — accent
COLOR_WHITE       = (255, 255, 255)
COLOR_GRAY        = (140, 140, 140)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_box_label(
    frame: np.ndarray,
    box: tuple,
    label: str,
    box_color: tuple,
    alpha: float = 0.7,
) -> None:
    """Vẽ bounding box và label có nền mờ."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    bg_y1 = max(0, y1 - th - 8)
    bg_y2 = y1

    # Nền label
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, bg_y1), (x1 + tw + 6, bg_y2), box_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.55, (0, 0, 0), 2)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.55, COLOR_WHITE, 1)


def _draw_sidebar(
    frame: np.ndarray,
    confirmed: Dict[int, str],
    fps: float,
    frame_idx: int,
    paused: bool,
) -> np.ndarray:
    """
    Vẽ sidebar bên phải hiển thị kết quả biển số đã chốt.
    Trả về frame đã có sidebar.
    """
    SIDEBAR_W = 260
    h, w = frame.shape[:2]

    # Tạo canvas mới rộng hơn để ghép sidebar
    canvas = np.zeros((h, w + SIDEBAR_W, 3), dtype=np.uint8)
    canvas[:, :w] = frame

    # Nền sidebar
    sidebar = canvas[:, w:]
    sidebar[:] = COLOR_PANEL_BG

    # Đường phân cách
    cv2.line(canvas, (w, 0), (w, h), COLOR_ACCENT, 1)

    x = w + 10
    y = 30

    # Tiêu đề
    cv2.putText(canvas, "RESULTS", (x, y), FONT, 0.65, COLOR_ACCENT, 2)
    y += 5
    cv2.line(canvas, (x, y), (x + SIDEBAR_W - 20, y), COLOR_ACCENT, 1)
    y += 20

    # FPS + Frame
    status = "PAUSED" if paused else f"FPS: {fps:.1f}"
    status_color = (0, 100, 255) if paused else (100, 255, 100)
    cv2.putText(canvas, status, (x, y), FONT, 0.5, status_color, 1)
    y += 18
    cv2.putText(canvas, f"Frame: {frame_idx:>6}", (x, y), FONT, 0.45, COLOR_GRAY, 1)
    y += 25

    cv2.line(canvas, (x, y), (x + SIDEBAR_W - 20, y), (50, 50, 50), 1)
    y += 15

    # Bảng kết quả
    cv2.putText(canvas, "ID    PLATE", (x, y), FONT, 0.45, COLOR_GRAY, 1)
    y += 18

    if not confirmed:
        cv2.putText(canvas, "Chua co ket qua...", (x, y), FONT, 0.42, COLOR_GRAY, 1)
    else:
        for tid, plate in list(confirmed.items())[-12:]:  # hiển thị tối đa 12 dòng
            line_txt = f"#{tid:<4}  {plate}"
            cv2.putText(canvas, line_txt, (x, y), FONT, 0.5, COLOR_CONFIRMED, 1)
            y += 22
            if y > h - 30:
                break

    # Footer
    cv2.putText(canvas, "q=thoat p=pause f=full s=save r=reset",
                (x, h - 15), FONT, 0.32, (80, 80, 80), 1)

    return canvas


# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self):
        self._times.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ---------------------------------------------------------------------------
# Result Saver
# ---------------------------------------------------------------------------

def save_result(
    track_id:     int,
    plate_text:   str,
    vehicle_crop: Optional[np.ndarray],
    plate_img:    Optional[np.ndarray],
    conf:         float,
    frame_idx:    int,
    annotated:    Optional[np.ndarray] = None,
) -> str:
    """
    Lưu kết quả nhận diện đã chốt vào data/results/car-<BIENSỐ>/:
      vehicle.jpg  — ảnh xe (hoặc vùng context xung quanh biển)
      plate.jpg    — crop biển số
      frame.jpg    — frame đầy đủ đã annotate
      info.txt     — thông tin text
    Trả về đường dẫn folder để in log.
    """
    folder = os.path.join(RESULTS_DIR, f"car-{plate_text}")
    os.makedirs(folder, exist_ok=True)

    if vehicle_crop is not None and vehicle_crop.size > 0:
        cv2.imwrite(os.path.join(folder, "vehicle.jpg"), vehicle_crop)

    if plate_img is not None and plate_img.size > 0:
        cv2.imwrite(os.path.join(folder, "plate.jpg"), plate_img)

    if annotated is not None:
        cv2.imwrite(os.path.join(folder, "frame.jpg"), annotated)

    with open(os.path.join(folder, "info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Biển số:    {plate_text}\n")
        f.write(f"Track ID:   {track_id}\n")
        f.write(f"Confidence: {conf:.3f}\n")
        f.write(f"Frame:      {frame_idx}\n")
        f.write(f"Thời gian:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return folder


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source,
    skip: int       = SKIP_FRAMES,
    vehicle_conf: float = VEHICLE_CONF,
    min_area: int   = MIN_AREA,
    min_votes: int  = MIN_VOTES,
    vote_ratio: float = VOTE_RATIO,
    device: str     = "cpu",
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load models ----
    print("\n" + "="*55)
    print("  LICENSE PLATE RECOGNITION — Đang khởi tạo...")
    print("="*55)

    try:
        vehicle_detector = VehicleDetector(
            conf     = vehicle_conf,
            min_area = min_area,
            device   = device,
        )
        plate_reader = PlateReader(device=device)
        vote_manager = VoteManager(
            min_votes     = min_votes,
            vote_ratio    = vote_ratio,
            track_timeout = TRACK_TIMEOUT,
            ocr_interval  = OCR_INTERVAL,
            max_samples   = MAX_SAMPLES,
        )
        zone_json = "data/zone.json"
        if os.path.exists(zone_json):
            with open(zone_json, encoding="utf-8") as _f:
                _cfg = json.load(_f)
            zone_filter = ZoneFilter.from_config(_cfg)
            pts = _cfg.get("points", [])
            print(f"[INFO] Zone loaded: {zone_json}  ({len(pts)} điểm)")
        else:
            zone_filter = ZoneFilter(top_ratio=ZONE_TOP, bottom_ratio=ZONE_BOTTOM)
            print(f"[INFO] Zone: toàn frame (chưa có data/zone.json — chạy tools/draw_zone.py để vẽ)")
    except FileNotFoundError as e:
        print(f"[LỖI] {e}")
        sys.exit(1)

    print(f"\n[INFO] Nguồn: {source}")
    print(f"[INFO] Frame skip: mỗi {skip} frame")
    print(f"[INFO] Phím: q=thoát  p=pause  f=fullscreen  s=lưu  r=reset\n")

    # ---- Mở video source ----
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được nguồn: {source}")
        sys.exit(1)

    # Cài đặt buffer nhỏ để giảm độ trễ RTSP
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # ---- Tính kích thước cửa sổ ----
    try:
        import tkinter as tk
        root = tk.Tk()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        sw, sh = 1920, 1080

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    SIDEBAR_W = 260
    scale     = min(sw * 0.85 / (vw + SIDEBAR_W), sh * 0.85 / vh)
    win_w     = int((vw + SIDEBAR_W) * scale)
    win_h     = int(vh * scale)

    WIN = "License Plate Recognition"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, win_w, win_h)
    cv2.moveWindow(WIN, (sw - win_w) // 2, (sh - win_h) // 2)

    # ---- Trạng thái ----
    fps_counter = FPSCounter()
    paused      = False
    fullscreen  = False
    frame_idx   = 0
    last_annotated = None

    # Cache: frame hiện tại đã xử lý
    confirmed_snapshot: Dict[int, str] = {}

    # Dedup: tránh lưu cùng biển số nhiều lần khi ByteTrack đổi ID
    # plate_text → frame_idx của lần chốt cuối cùng
    plate_last_confirmed: Dict[str, int] = {}
    DEDUP_FRAMES = 300  # bỏ qua nếu cùng biển đã chốt trong vòng 300 frame

    print("[INFO] Pipeline đang chạy...\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Hết video / mất kết nối stream.")
                break
            frame_idx += 1

            # ---- Frame Skipping ----
            if frame_idx % skip != 0:
                # Vẫn hiển thị frame cũ để video không bị giật
                if last_annotated is not None:
                    cv2.imshow(WIN, last_annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                continue

            # ---- Stage 1+2: Vehicle Detection & Tracking ----
            vehicles = vehicle_detector.track(frame)

            # Fallback cho camera góc từ trên (top-down): nếu không detect được xe,
            # tìm biển số trực tiếp trên toàn frame bằng plate-detect model.
            plate_direct = False
            if not vehicles:
                vehicles = plate_reader.track_plates_on_frame(frame)
                plate_direct = bool(vehicles)

            # Dọn dẹp các track đã đi
            removed = vote_manager.cleanup(frame_idx)
            if removed:
                for tid in removed:
                    print(f"  [TRACK] ID {tid} đã rời khỏi màn hình")

            annotated = frame.copy()

            # Vẽ Zone of Interest
            zone_filter.draw(annotated)

            for v in vehicles:
                vote_manager.update_seen(v.track_id, frame_idx)

                state = vote_manager.get_state(v.track_id)
                confirmed_plate = vote_manager.get_confirmed(v.track_id)

                # Màu box phụ thuộc vào trạng thái
                if confirmed_plate:
                    box_color = COLOR_CONFIRMED
                elif state == VehicleState.SAMPLING:
                    box_color = COLOR_PLATE   # Vàng — đang sampling
                else:
                    box_color = COLOR_VEHICLE  # Xanh — đang detecting

                # Label cho xe
                plate_display = confirmed_plate or ("[OCR]" if state == VehicleState.SAMPLING else "...")
                label = f"{v.class_name}#{v.track_id}  {plate_display}"
                _draw_box_label(annotated, v.box, label, box_color)

                # ---- State Machine ----
                if state == VehicleState.DONE:
                    # Đã chốt — không làm gì thêm
                    continue

                # Plate-direct mode: bỏ qua zone filter (toàn frame đều hợp lệ)
                in_zone = plate_direct or zone_filter.is_in_zone(v.box, frame.shape)

                if state == VehicleState.DETECTING:
                    # Chuyển sang SAMPLING khi xe vào zone
                    if in_zone:
                        vote_manager.transition_to_sampling(v.track_id)
                        # Cập nhật state ngay để có thể OCR trong frame này
                        state = vote_manager.get_state(v.track_id)

                if state == VehicleState.SAMPLING:
                    if not in_zone:
                        # Xe tạm ra khỏi zone — bỏ qua frame này
                        continue

                    # ---- Per-vehicle sampling rate control ----
                    # Plate-direct mode: OCR mỗi frame vì xe qua nhanh
                    if not plate_direct and not vote_manager.should_run_ocr(v.track_id, frame_idx):
                        continue

                    # ---- Stage 3+4+5: Plate Detection + Warp + OCR ----
                    if plate_direct:
                        plate_res = plate_reader.read_direct_plate(
                            frame, v.box, track_id=v.track_id
                        )
                    else:
                        plate_res = plate_reader.read(frame, v.box, track_id=v.track_id)

                    text = plate_res.text if plate_res else None
                    conf = plate_res.conf if plate_res else 0.0

                    if text and text != "Unknown":
                        norm  = normalize_plate(text)
                        valid = is_valid_vn_plate(norm)
                        summary = vote_manager.get_history_summary(v.track_id)
                        votes   = summary.get("total_reads", 0)
                        status  = "HOP LE" if valid else "SAI DINH DANG"
                        print(f"  [VOTE]          ID {v.track_id}: '{norm}' {status}  conf={conf:.2f}  votes={votes}/{vote_manager.min_votes}")

                    confirmed = vote_manager.record_ocr(
                        track_id  = v.track_id,
                        text      = text,
                        conf      = conf,
                        frame_idx = frame_idx,
                    )

                    if confirmed:
                        last_frame = plate_last_confirmed.get(confirmed, -DEDUP_FRAMES - 1)
                        is_dup = (frame_idx - last_frame) <= DEDUP_FRAMES
                        print(
                            f"\n  ✅ [CHỐT] ID={v.track_id} ({v.class_name})"
                            f"  Biển số: {confirmed}  conf={conf:.2f}"
                            + ("  [DUP — bỏ qua lưu]" if is_dup else "") + "\n"
                        )
                        plate_last_confirmed[confirmed] = frame_idx
                        # Lưu kết quả ra folder riêng (bỏ qua nếu trùng lặp gần đây)
                        if plate_res and not is_dup:
                            result_name = save_result(
                                track_id     = v.track_id,
                                plate_text   = confirmed,
                                vehicle_crop = plate_res.vehicle_crop,
                                plate_img    = plate_res.warped_img,
                                conf         = plate_res.conf,
                                frame_idx    = frame_idx,
                                annotated    = annotated,
                            )
                            print(f"  [LƯU] {result_name}/\n")
                        # Vẽ bbox biển số
                        if plate_res:
                            _draw_box_label(
                                annotated,
                                plate_res.plate_box,
                                f"{confirmed} ✓",
                                COLOR_CONFIRMED,
                            )
                    elif plate_res and plate_res.text != "Unknown":
                        # Chưa chốt — vẽ tạm kết quả hiện tại
                        _draw_box_label(
                            annotated,
                            plate_res.plate_box,
                            f"{plate_res.text} ({plate_res.conf:.2f})",
                            COLOR_PLATE,
                        )

            # Cập nhật confirmed snapshot
            confirmed_snapshot = vote_manager.all_confirmed()

            # FPS
            fps_counter.tick()

            # ---- Render sidebar ----
            canvas = _draw_sidebar(
                annotated,
                confirmed_snapshot,
                fps_counter.fps,
                frame_idx,
                paused,
            )

            last_annotated = canvas
            cv2.imshow(WIN, canvas)

        else:
            # Đang pause: hiển thị frame dừng
            if last_annotated is not None:
                pause_frame = last_annotated.copy()
                h, w = pause_frame.shape[:2]
                cv2.putText(
                    pause_frame, "[ PAUSED ]",
                    (w // 2 - 80, h // 2),
                    FONT, 1.2, (0, 100, 255), 3,
                )
                cv2.imshow(WIN, pause_frame)

        # ---- Xử lý phím bấm ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('p'):
            paused = not paused
            print(f"[INFO] {'Tạm dừng' if paused else 'Tiếp tục'}")

        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

        elif key == ord('s') and last_annotated is not None:
            save_path = os.path.join(RESULTS_DIR, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(save_path, last_annotated)
            print(f"  [SAVE] {save_path}")

        elif key == ord('r'):
            vehicle_detector.reset()
            vote_manager._records.clear()
            confirmed_snapshot.clear()
            frame_idx = 0
            print("[INFO] Đã reset tracker và voting data")

    # ---- Cleanup ----
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("  KẾT QUẢ CUỐI CÙNG")
    print("="*55)
    final = vote_manager.all_confirmed()
    if final:
        for tid, plate in sorted(final.items()):
            summary = vote_manager.get_history_summary(tid)
            votes = summary.get("total_reads", 0)
            print(f"  ID {tid:>3}: {plate}  ({votes} lần đọc)")
    else:
        print("  Không có biển số nào được chốt.")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_source(raw: str):
    """Chuyển đổi source string: số → int (webcam), còn lại giữ nguyên."""
    try:
        return int(raw)
    except ValueError:
        return raw


def main():
    parser = argparse.ArgumentParser(
        description="Nhận diện biển số xe thời gian thực",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python main.py --source 0                          # Webcam đầu tiên
  python main.py --source video.mp4 --skip 2        # File video, skip mỗi 2 frame
  python main.py --source rtsp://admin:admin@192.168.1.100:554/stream1
        """
    )

    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Nguồn video: số thiết bị (0), file (.mp4), hoặc RTSP URL",
    )
    parser.add_argument(
        "--skip", "-k",
        type=int,
        default=SKIP_FRAMES,
        help=f"Xử lý 1 trong mỗi N frames (default: {SKIP_FRAMES})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=VEHICLE_CONF,
        help=f"Ngưỡng confidence vehicle detection (default: {VEHICLE_CONF})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=MIN_AREA,
        help=f"Diện tích xe tối thiểu để xử lý (default: {MIN_AREA} px²)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=MIN_VOTES,
        help=f"Số frame đọc tối thiểu để chốt biển số (default: {MIN_VOTES})",
    )
    parser.add_argument(
        "--vote-ratio",
        type=float,
        default=VOTE_RATIO,
        help=f"Tỉ lệ đồng thuận để chốt (default: {VOTE_RATIO})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "0"],
        help="Thiết bị tính toán (default: cpu)",
    )

    args = parser.parse_args()
    source = _parse_source(args.source)

    run_pipeline(
        source     = source,
        skip       = args.skip,
        vehicle_conf = args.conf,
        min_area   = args.min_area,
        min_votes  = args.min_votes,
        vote_ratio = args.vote_ratio,
        device     = args.device,
    )


if __name__ == "__main__":
    main()
