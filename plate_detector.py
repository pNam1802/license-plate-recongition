"""
Nhận diện và đọc biển số xe từ ảnh hoặc video.

Cách dùng:
    python plate_detector.py image.jpg
    python plate_detector.py video.mp4

Phím tắt (chế độ video):
    q  - Thoát
    p  - Tạm dừng / Tiếp tục
    s  - Lưu frame hiện tại ra data/debug/
"""

import sys
import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

MODEL_PATH = "models/plate-detect.pt"
CONF       = 0.25
ALLOWLIST  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_plate_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    print("[INFO] Đang tải plate model từ HuggingFace...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="Koushim/yolov8-license-plate-detection",
        filename="best.pt",
        local_dir="models",
    )
    return YOLO(path)


# ---------------------------------------------------------------------------
# Preprocessing + OCR
# ---------------------------------------------------------------------------

def preprocess(plate_img):
    h, w = plate_img.shape[:2]
    if h < 100:
        scale     = max(2.0, 150 / h)
        plate_img = cv2.resize(plate_img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)


def read_plate(reader: easyocr.Reader, plate_img) -> str:
    processed = preprocess(plate_img)
    results   = reader.readtext(processed, allowlist=ALLOWLIST, detail=1)
    texts     = [t for (_, t, conf) in results if conf > 0.2 and t.strip()]
    return "".join(texts).upper().strip() or "Unknown"


# ---------------------------------------------------------------------------
# Detection on a single frame
# ---------------------------------------------------------------------------

def detect_frame(frame, model: YOLO, reader: easyocr.Reader):
    """Trả về frame đã annotate và list kết quả [(box, plate_text, conf)]."""
    results  = model(frame, conf=CONF, verbose=False)[0]
    findings = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_score      = float(box.conf[0])

        plate_crop = frame[max(0, y1):y2, max(0, x1):x2]
        if plate_crop.size == 0:
            continue

        plate_text = read_plate(reader, plate_crop)
        findings.append(((x1, y1, x2, y2), plate_text, conf_score))

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{plate_text}  {conf_score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, findings


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def run_image(path, model, reader):
    img = cv2.imread(path)
    if img is None:
        print(f"[LỖI] Không đọc được ảnh: {path}")
        sys.exit(1)

    annotated, findings = detect_frame(img, model, reader)

    print(f"\nKết quả ({len(findings)} biển số):")
    for (x1, y1, x2, y2), text, conf in findings:
        print(f"  {text:<15} conf={conf:.2f}  box=({x1},{y1})-({x2},{y2})")

    out_path = os.path.join("data/debug", f"result_{os.path.basename(path)}")
    cv2.imwrite(out_path, annotated)
    print(f"Ảnh kết quả: {out_path}")

    cv2.namedWindow("Plate Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Plate Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

def run_video(path, model, reader):
    import tkinter as tk
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {path}")
        sys.exit(1)

    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale  = min(sw * 0.85 / vw, sh * 0.85 / vh)
    win_w, win_h = int(vw * scale), int(vh * scale)

    WIN = "Plate Detection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, win_w, win_h)
    cv2.moveWindow(WIN, (sw - win_w) // 2, (sh - win_h) // 2)

    paused     = False
    fullscreen = False
    frame_idx  = 0
    print("[INFO] q=thoát  p=tạm dừng  f=fullscreen  s=lưu frame")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            annotated, findings = detect_frame(frame.copy(), model, reader)
            for _, text, conf in findings:
                print(f"  frame={frame_idx:>5}  {text:<15} conf={conf:.2f}")

            cv2.imshow(WIN, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        elif key == ord('s') and not paused:
            out = f"data/debug/frame_{frame_idx}.jpg"
            cv2.imwrite(out, annotated)
            print(f"  [SAVE] {out}")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python plate_detector.py <ảnh hoặc video>")
        sys.exit(1)

    input_path = sys.argv[1]
    print("[INFO] Đang tải models...")
    model  = load_plate_model()
    reader = easyocr.Reader(['en'], gpu=False)
    print("[INFO] Sẵn sàng.\n")

    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        run_image(input_path, model, reader)
    else:
        run_video(input_path, model, reader)
