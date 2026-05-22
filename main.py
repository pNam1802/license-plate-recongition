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
from pipeline.zone_filter import LineFilter


# ---------------------------------------------------------------------------
# Config mặc định
# ---------------------------------------------------------------------------

SKIP_FRAMES   = 2        # Xử lý 1 frame trong mỗi N frames
MIN_AREA      = 4000     # Diện tích xe tối thiểu (px²)
VEHICLE_CONF  = 0.35     # Confidence ngưỡng vehicle detection
TRACK_TIMEOUT = 90       # Frames không thấy thì xóa track
MIN_VOTES     = 3        # Số lần đọc tối thiểu để chốt biển số
VOTE_RATIO    = 0.55     # Tỉ lệ đồng thuận để chốt
OCR_INTERVAL  = 3        # Số frame giữa 2 lần OCR cho cùng 1 xe (ảnh liên tiếp gần giống nhau)
MAX_SAMPLES   = 30       # Số lần OCR tối đa trước khi bỏ qua xe
RESULTS_DIR   = "data/results"

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


def _draw_zone_counter(frame: np.ndarray, count: int) -> None:
    """Vẽ bộ đếm xe đã vượt đường kẻ góc trên-trái."""
    label = f"COUNT: {count}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.75, 2)
    pad = 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, label, (10 + pad, 10 + pad + th),
                FONT, 0.75, (60, 230, 60), 2)


def _draw_sidebar(
    frame: np.ndarray,
    confirmed: Dict[int, str],
    fps: float,
    frame_idx: int,
    paused: bool,
) -> np.ndarray:
    """
    Vẽ panel kết quả dạng overlay trong suốt góc trên-phải của frame.
    """
    PANEL_W   = 300
    PANEL_PAD = 12
    LINE_H    = 32   # khoảng cách dòng (chữ to hơn)
    FONT_SCALE_TITLE  = 0.75
    FONT_SCALE_RESULT = 0.72
    FONT_SCALE_INFO   = 0.55
    COLOR_RESULT = (60, 230, 60)    # xanh lá sáng
    COLOR_TITLE  = (60, 230, 60)
    COLOR_INFO   = (140, 220, 140)  # xanh lá nhạt

    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Tính chiều cao panel theo số dòng kết quả
    n_results = max(1, len(confirmed))
    n_rows    = min(n_results, 10)
    panel_h   = PANEL_PAD + 30 + 24 + 8 + LINE_H * n_rows + PANEL_PAD

    x0 = w - PANEL_W - 10
    y0 = 10
    x1 = w - 10
    y1 = y0 + panel_h

    # Nền trong suốt
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # Viền mỏng
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_TITLE, 1)

    x = x0 + PANEL_PAD
    y = y0 + PANEL_PAD + 18

    # Tiêu đề
    cv2.putText(canvas, "RESULTS", (x, y), FONT, FONT_SCALE_TITLE, COLOR_TITLE, 2)
    y += 6
    cv2.line(canvas, (x, y), (x1 - PANEL_PAD, y), COLOR_TITLE, 1)
    y += 22

    # FPS
    if paused:
        cv2.putText(canvas, "[ PAUSED ]", (x, y), FONT, FONT_SCALE_INFO, (60, 120, 255), 1)
    else:
        cv2.putText(canvas, f"FPS: {fps:.1f}   Frame: {frame_idx}", (x, y),
                    FONT, FONT_SCALE_INFO, COLOR_INFO, 1)
    y += LINE_H - 4

    # Danh sách kết quả
    if not confirmed:
        cv2.putText(canvas, "Chua co ket qua...", (x, y), FONT, FONT_SCALE_INFO, COLOR_INFO, 1)
    else:
        for tid, plate in list(confirmed.items())[-10:]:
            cv2.putText(canvas, f"#{tid:<3} {plate}", (x, y), FONT, FONT_SCALE_RESULT,
                        COLOR_RESULT, 2)
            y += LINE_H
            if y > y1 - PANEL_PAD:
                break

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
    skip: int         = SKIP_FRAMES,
    vehicle_conf: float = VEHICLE_CONF,
    min_area: int     = MIN_AREA,
    min_votes: int    = MIN_VOTES,
    vote_ratio: float = VOTE_RATIO,
    device: str       = "cpu",
    validate: bool    = True,
    save_img: bool    = True,
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
        plate_reader = PlateReader(device=device, validate=validate)
        vote_manager = VoteManager(
            min_votes     = min_votes,
            vote_ratio    = vote_ratio,
            track_timeout = TRACK_TIMEOUT,
            ocr_interval  = OCR_INTERVAL,
            max_samples   = MAX_SAMPLES,
            validate      = validate,
        )
        if not validate:
            print("[INFO] Chế độ: KHÔNG validate định dạng biển số (--no-validate)")
        line_json = "data/line.json"
        if os.path.exists(line_json):
            with open(line_json, encoding="utf-8") as _f:
                _cfg = json.load(_f)
            line_filter = LineFilter.from_config(_cfg)
            print(f"[INFO] Line loaded: {line_json}  P1={line_filter.p1}  P2={line_filter.p2}")
        else:
            line_filter = None
            print("[INFO] Không có line (chạy tools/draw_line.py để vẽ) — xe sẽ không bị đếm")
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
    scale = min(sw * 0.85 / vw, sh * 0.85 / vh)
    win_w = int(vw * scale)
    win_h = int(vh * scale)

    WIN = "License Plate Recognition"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, win_w, win_h)
    cv2.moveWindow(WIN, (sw - win_w) // 2, (sh - win_h) // 2)

    # ---- Video writer ----
    video_writer = None
    if save_img:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(RESULTS_DIR, f"output_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        video_writer = cv2.VideoWriter(out_path, fourcc, src_fps / max(1, skip), (vw, vh))
        print(f"[INFO] Lưu video output: {out_path}")

    # ---- Trạng thái ----
    fps_counter = FPSCounter()
    paused      = False
    fullscreen  = False
    frame_idx   = 0
    last_annotated = None

    # Tích lũy toàn bộ biển số đã chốt trong session — không bao giờ tự xóa
    session_confirmed: Dict[int, str] = {}
    confirmed_snapshot: Dict[int, str] = {}

    # Dedup: tránh lưu cùng biển số nhiều lần khi ByteTrack đổi ID
    # plate_text → frame_idx của lần chốt cuối cùng
    plate_last_confirmed: Dict[str, int] = {}
    DEDUP_FRAMES = 300  # bỏ qua nếu cùng biển đã chốt trong vòng 300 frame

    # Zone counter: đếm xe đi vào zone (mỗi track_id chỉ đếm 1 lần)
    zone_entered_ids: set = set()
    zone_count: int = 0

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
                    if line_filter is not None:
                        line_filter.remove_track(tid)

            annotated = frame.copy()

            # Vẽ đường kẻ
            if line_filter is not None:
                line_filter.draw(annotated)

            # Vẽ line counter góc trên-trái
            _draw_zone_counter(annotated, zone_count)

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

                # Luôn gọi check_crossing để cập nhật _prev_side (theo dõi vị trí)
                crossed = (
                    line_filter is not None
                    and line_filter.check_crossing(v.track_id, v.box)
                )

                if state == VehicleState.DETECTING:
                    if line_filter is None:
                        # Không có line → bắt đầu OCR ngay
                        vote_manager.transition_to_sampling(v.track_id)
                        state = vote_manager.get_state(v.track_id)
                    elif crossed:
                        # Vượt đường → đếm + bắt đầu OCR
                        if v.track_id not in zone_entered_ids:
                            zone_entered_ids.add(v.track_id)
                            zone_count += 1
                            print(f"  [LINE] Xe #{v.track_id} vượt đường — tổng: {zone_count}")
                        vote_manager.transition_to_sampling(v.track_id)
                        state = vote_manager.get_state(v.track_id)
                    else:
                        # Chưa vượt đường → chờ, không OCR
                        continue

                if state == VehicleState.SAMPLING:
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
                        session_confirmed[v.track_id] = confirmed
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
            confirmed_snapshot = session_confirmed

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
            if video_writer is not None:
                video_writer.write(annotated)

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
            zone_entered_ids.clear()
            zone_count = 0
            session_confirmed.clear()
            confirmed_snapshot.clear()
            frame_idx = 0
            print("[INFO] Đã reset tracker, voting data và zone counter")

    # ---- Cleanup ----
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"[INFO] Video đã lưu: {out_path}")
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
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Bỏ qua regex validation biển số VN — dùng khi test biển nước ngoài",
    )
    parser.add_argument(
        "--save-img",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="True|False",
        help="Lưu video output vào data/results/ (default: True)",
    )

    args = parser.parse_args()
    source = _parse_source(args.source)

    run_pipeline(
        source       = source,
        skip         = args.skip,
        vehicle_conf = args.conf,
        min_area     = args.min_area,
        min_votes    = args.min_votes,
        vote_ratio   = args.vote_ratio,
        device       = args.device,
        validate     = not args.no_validate,
        save_img     = args.save_img,
    )


if __name__ == "__main__":
    main()
