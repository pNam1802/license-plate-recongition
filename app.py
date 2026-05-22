"""
app.py — Streamlit Web UI cho hệ thống nhận diện biển số xe
=============================================================
Tính năng:
  - Upload file video .mp4
  - Vẽ Zone of Interest bằng chuột trên frame đầu tiên
  - Pipeline nhận diện chạy trong background thread
  - Cột phải: danh sách biển số (đã chốt / chưa chốt)
  - Click vào biển số → xem ảnh biển + ảnh xe

Chạy:
    streamlit run app.py
"""

import os
import sys
import threading
import time
import queue
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Cấu hình trang
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nhận Diện Biển Số Xe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS tùy chỉnh
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Ẩn header mặc định của Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.app-header h1 {
    color: white;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    color: #a0a8c8;
    font-size: 0.85rem;
    margin: 2px 0 0 0;
}

/* Thẻ biển số */
.plate-card {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.plate-card:hover {
    border-color: #4a9eff;
    background: #1e1e3a;
    transform: translateX(3px);
}
.plate-card.confirmed {
    border-left: 4px solid #22c55e;
}
.plate-card.pending {
    border-left: 4px solid #f59e0b;
}
.plate-number {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 2px;
    font-family: 'Courier New', monospace;
}
.plate-meta {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
}
.badge-confirmed {
    background: #166534;
    color: #86efac;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
}
.badge-pending {
    background: #78350f;
    color: #fcd34d;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Stats */
.stat-box {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4a9eff;
}
.stat-label {
    font-size: 0.72rem;
    color: #64748b;
}

/* Video placeholder */
.video-placeholder {
    background: #0f0f1a;
    border: 2px dashed #2d2d44;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 300px;
    color: #4a5568;
    font-size: 0.9rem;
}

/* Zone instruction */
.zone-info {
    background: #1e3a5f;
    border: 1px solid #2563eb;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #93c5fd;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """Lưu một kết quả nhận diện biển số."""
    track_id:     int
    plate_text:   str
    conf:         float
    confirmed:    bool
    vehicle_img:  Optional[np.ndarray] = field(default=None, repr=False)
    plate_img:    Optional[np.ndarray] = field(default=None, repr=False)
    timestamp:    float = field(default_factory=time.time)
    class_name:   str = "vehicle"

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "running":        False,
        "stop_event":     threading.Event(),
        "video_path":     None,
        "uploaded_name":  None,
        "first_frame":    None,
        "zone_polygon":   None,
        "current_frame":  None,
        "detections":     {},
        "selected_id":    None,
        "frame_count":    0,
        "fps_display":    0.0,
        "thread":         None,
        "frame_queue":    queue.Queue(maxsize=2),
        "detection_queue":queue.Queue(maxsize=100),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def pipeline_thread(
    video_path: str,
    zone_polygon,
    frame_q: queue.Queue,
    det_q: queue.Queue,
    stop_event: threading.Event,
    min_votes: int = 3,
    ocr_interval: int = 5,
    skip: int = 2,
):
    """
    Chạy toàn bộ pipeline nhận diện trong background thread.
    Đẩy frame đã annotate vào frame_q, kết quả vào det_q.
    """
    try:
        from pipeline.vehicle_detector import VehicleDetector
        from pipeline.plate_reader import PlateReader
        from pipeline.vote_logic import VoteManager, VehicleState
        from pipeline.zone_filter import ZoneFilter

        vehicle_detector = VehicleDetector(conf=0.40, min_area=4000)
        plate_reader     = PlateReader()
        vote_manager     = VoteManager(
            min_votes=min_votes, vote_ratio=0.5,
            track_timeout=90, ocr_interval=ocr_interval, max_samples=40,
        )

        # Khởi tạo ZoneFilter
        if zone_polygon and len(zone_polygon) >= 3:
            zone_filter = ZoneFilter(polygon=zone_polygon)
        else:
            zone_filter = ZoneFilter(top_ratio=0.20, bottom_ratio=0.85)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            det_q.put({"error": f"Không mở được video: {video_path}"})
            return

        frame_idx   = 0
        fps_counter = []
        SKIP        = skip
        COLOR_CONFIRMED = (50,  255, 50)
        COLOR_PLATE     = (0,   220, 255)
        COLOR_VEHICLE   = (0,   200, 50)
        COLOR_SAMPLING  = (0,   220, 255)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            fps_counter.append(time.time())
            if len(fps_counter) > 30:
                fps_counter.pop(0)

            if frame_idx % SKIP != 0:
                continue

            annotated = frame.copy()
            zone_filter.draw(annotated)

            vehicles = vehicle_detector.track(frame)
            removed  = vote_manager.cleanup(frame_idx)

            for v in vehicles:
                vote_manager.update_seen(v.track_id, frame_idx)
                state = vote_manager.get_state(v.track_id)
                confirmed_plate = vote_manager.get_confirmed(v.track_id)

                # Màu box
                if confirmed_plate:
                    box_color = COLOR_CONFIRMED
                elif state == VehicleState.SAMPLING:
                    box_color = COLOR_SAMPLING
                else:
                    box_color = COLOR_VEHICLE

                # Vẽ bounding box xe
                x1, y1, x2, y2 = v.box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                plate_disp = confirmed_plate or ("[OCR]" if state == VehicleState.SAMPLING else "...")
                lbl = f"#{v.track_id} {v.class_name} {plate_disp}"
                (tw, th), _ = cv2.getTextSize(lbl, FONT, 0.5, 1)
                cv2.rectangle(annotated, (x1, max(0, y1-th-8)), (x1+tw+6, y1), box_color, -1)
                cv2.putText(annotated, lbl, (x1+3, y1-4), FONT, 0.5, (0,0,0), 2)
                cv2.putText(annotated, lbl, (x1+3, y1-4), FONT, 0.5, (255,255,255), 1)

                # ---- State Machine ----
                if state == VehicleState.DONE:
                    continue

                in_zone = zone_filter.is_in_zone(v.box, frame.shape)
                if state == VehicleState.DETECTING:
                    if in_zone:
                        vote_manager.transition_to_sampling(v.track_id)
                elif state == VehicleState.SAMPLING:
                    if not in_zone:
                        continue
                    if not vote_manager.should_run_ocr(v.track_id, frame_idx):
                        continue

                    # Crop xe để lưu ảnh
                    vx1, vy1, vx2, vy2 = v.box
                    fh, fw = frame.shape[:2]
                    vx1 = max(0, vx1); vy1 = max(0, vy1)
                    vx2 = min(fw, vx2); vy2 = min(fh, vy2)
                    vehicle_crop_img = frame[vy1:vy2, vx1:vx2].copy()

                    plate_res = plate_reader.read(frame, v.box, track_id=v.track_id)

                    text = plate_res.text if plate_res else None
                    conf = plate_res.conf if plate_res else 0.0

                    # Gửi detection (kể cả chưa chốt) lên UI
                    if text and text != "Unknown":
                        # Vẽ bbox biển số lên frame
                        if plate_res:
                            pb = plate_res.plate_box
                            cv2.rectangle(annotated, (pb[0], pb[1]), (pb[2], pb[3]), COLOR_PLATE, 2)
                            cv2.putText(annotated, f"{text} ({conf:.2f})",
                                       (pb[0], pb[1]-5), FONT, 0.5, COLOR_PLATE, 1)

                        confirmed = vote_manager.record_ocr(
                            track_id=v.track_id, text=text, conf=conf, frame_idx=frame_idx
                        )
                        is_confirmed = confirmed is not None

                        det_q.put({
                            "track_id":    v.track_id,
                            "plate_text":  confirmed or text,
                            "conf":        conf,
                            "confirmed":   is_confirmed,
                            "vehicle_img": vehicle_crop_img,
                            "plate_img":   plate_res.warped_img.copy() if plate_res and plate_res.warped_img is not None else None,
                            "class_name":  v.class_name,
                        })
                    else:
                        vote_manager.mark_ocr_attempted(v.track_id, frame_idx)

            # Tính FPS
            fps = 0.0
            if len(fps_counter) >= 2:
                fps = (len(fps_counter) - 1) / (fps_counter[-1] - fps_counter[0])

            # Đẩy frame ra UI (không block nếu queue đầy)
            try:
                frame_q.put_nowait({
                    "frame": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    "fps":   fps,
                    "idx":   frame_idx,
                })
            except queue.Full:
                pass

        cap.release()
        det_q.put({"done": True})

    except Exception as e:
        import traceback
        det_q.put({"error": str(e), "tb": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Custom Zone Drawing Component (HTML/JS Canvas)
# ---------------------------------------------------------------------------

def _frame_to_b64(frame_bgr) -> str:
    """Encode numpy BGR frame thành base64 JPEG string."""
    import base64
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def zone_canvas_component(frame_bgr, existing_points=None):
    """
    Hiển thị custom HTML/JS canvas cho phép vẽ polygon zone.
    - Click để thêm điểm
    - Double-click để đóng polygon
    - Kéo điểm để chỉnh lại sau khi đóng
    Trả về list[(x,y)] trong tọa độ frame gốc, hoặc None.
    """
    import json
    import streamlit.components.v1 as components

    orig_h, orig_w = frame_bgr.shape[:2]
    CANVAS_W = 700
    CANVAS_H = int(CANVAS_W * orig_h / orig_w)
    img_b64  = _frame_to_b64(frame_bgr)
    init_pts = json.dumps(existing_points or [])

    html_code = f"""
    <style>
    #zone-canvas  {{ border: 2px solid #f59e0b; border-radius: 8px; cursor: crosshair; display: block; }}
    #zone-hint    {{ font-family: monospace; font-size: 12px; color: #93c5fd; margin-top: 6px; }}
    #zone-output  {{ width: 100%; font-size: 11px; margin-top: 4px; background:#1a1a2e;
                    color:#e2e8f0; border:1px solid #2d2d44; border-radius:4px; padding:4px; }}
    #btn-clear    {{ margin-top:6px; padding:4px 14px; background:#7f1d1d; color:#fca5a5;
                    border:none; border-radius:4px; cursor:pointer; font-size:12px; }}
    #btn-clear:hover {{ background:#991b1b; }}
    </style>
    <div>
      <canvas id="zone-canvas" width="{CANVAS_W}" height="{CANVAS_H}"></canvas>
      <div id="zone-hint">🖱 Click để thêm điểm &middot; Double-click để đóng zone &middot; Kéo điểm để chỉnh lại</div>
      <button id="btn-clear">🗑 Xóa zone</button><br/>
      <small style="color:#64748b">Sau khi vẽ xong, copy chuỗi JSON bên dưới vào ô nhập bên dưới canvas:</small><br/>
      <textarea id="zone-output" rows="2" readonly></textarea>
    </div>
    <script>
    (function() {{
      const canvas = document.getElementById('zone-canvas');
      const ctx    = canvas.getContext('2d');
      const output = document.getElementById('zone-output');
      const btnClear = document.getElementById('btn-clear');
      const scaleX = {orig_w} / {CANVAS_W};
      const scaleY = {orig_h} / {CANVAS_H};
      const img = new Image();
      img.src = 'data:image/jpeg;base64,{img_b64}';
      let points  = {init_pts};
      let closed  = points.length >= 3;
      let dragIdx = -1;
      const R = 7;

      function draw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        if (!points.length) return;
        if (closed) {{
          ctx.beginPath();
          ctx.moveTo(points[0][0], points[0][1]);
          points.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
          ctx.closePath();
          ctx.fillStyle = 'rgba(245,158,11,0.18)';
          ctx.fill();
        }}
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        points.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
        if (closed) ctx.closePath();
        ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2.5; ctx.stroke();
        points.forEach((p, i) => {{
          ctx.beginPath(); ctx.arc(p[0], p[1], R, 0, Math.PI*2);
          ctx.fillStyle = i === 0 ? '#22c55e' : '#f59e0b'; ctx.fill();
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
        }});
      }}

      function pos(e) {{
        const r = canvas.getBoundingClientRect();
        return [e.clientX - r.left, e.clientY - r.top];
      }}
      function nearIdx(px, py) {{
        return points.findIndex(p => Math.hypot(p[0]-px, p[1]-py) < R+4);
      }}
      function emit() {{
        const fp = points.map(p => [Math.round(p[0]*scaleX), Math.round(p[1]*scaleY)]);
        output.value = JSON.stringify(fp);
      }}

      canvas.addEventListener('click', e => {{
        if (closed) return;
        points.push(pos(e)); draw(); emit();
      }});
      canvas.addEventListener('dblclick', e => {{
        if (points.length < 3) return;
        points.pop(); closed = true; draw(); emit();
      }});
      canvas.addEventListener('mousedown', e => {{
        if (!closed) return;
        const [px, py] = pos(e); dragIdx = nearIdx(px, py);
      }});
      canvas.addEventListener('mousemove', e => {{
        if (dragIdx < 0) return;
        points[dragIdx] = pos(e); draw();
      }});
      canvas.addEventListener('mouseup', () => {{
        if (dragIdx >= 0) {{ dragIdx = -1; emit(); }}
      }});
      btnClear.addEventListener('click', () => {{
        points = []; closed = false; draw(); output.value = '[]';
      }});
      img.onload = () => draw();
      if (img.complete) draw();
    }})();
    </script>
    """

    components.html(html_code, height=CANVAS_H + 110, scrolling=False)

    # Nhận tọa độ qua text_input: user copy JSON từ canvas vào ô này
    raw = st.text_input(
        "Paste JSON tọa độ zone (copy từ ô canvas):",
        key="zone_pts_raw",
        placeholder='Ví dụ: [[100,200],[500,200],[500,400],[100,400]]',
        help="Copy chuỗi JSON từ ô textarea trên canvas vào đây rồi nhấn Enter",
    )
    if raw and raw.strip() not in ("", "[]"):
        try:
            import json as _json
            pts = _json.loads(raw)
            if isinstance(pts, list) and len(pts) >= 3:
                return [(int(p[0]), int(p[1])) for p in pts]
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="app-header">
        <div>
            <h1>🚗 Nhận Diện Biển Số Xe</h1>
            <p>Hệ thống ANPR thời gian thực — State Machine · Zone of Interest · Voting</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_detection_card(rec: DetectionRecord, is_selected: bool) -> bool:
    """Vẽ thẻ biển số, trả về True nếu được click."""
    status_class  = "confirmed" if rec.confirmed else "pending"
    badge_class   = "badge-confirmed" if rec.confirmed else "badge-pending"
    badge_text    = "✓ Đã chốt" if rec.confirmed else "⏳ Chờ"
    conf_text     = f"{rec.conf*100:.0f}%"
    time_text     = time.strftime("%H:%M:%S", time.localtime(rec.timestamp))

    selected_style = "border-color: #4a9eff; background: #1e2d4a;" if is_selected else ""

    st.markdown(f"""
    <div class="plate-card {status_class}" style="{selected_style}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="plate-number">{rec.plate_text}</span>
            <span class="{badge_class}">{badge_text}</span>
        </div>
        <div class="plate-meta">
            🚗 {rec.class_name} · ID #{rec.track_id} · conf {conf_text} · {time_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.button("🔍 Xem ảnh", key=f"btn_{rec.track_id}_{rec.plate_text}", use_container_width=True)


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Chuyển numpy BGR → PIL RGB."""
    if img is None:
        return None
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    render_header()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")

        uploaded = st.file_uploader(
            "📁 Upload video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Chọn file video để nhận diện biển số",
        )

        if uploaded:
            # Lưu file tạm khi upload file mới
            if st.session_state.uploaded_name != uploaded.name:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded.name)[1]
                )
                tmp.write(uploaded.read())
                tmp.close()
                st.session_state.video_path     = tmp.name
                st.session_state.uploaded_name  = uploaded.name
                st.session_state.first_frame    = None
                st.session_state.zone_polygon   = None
                st.session_state.running        = False
                st.session_state.detections     = {}
                st.session_state.selected_id    = None
                if "zone_pts_raw" in st.session_state:
                    del st.session_state["zone_pts_raw"]

            # Đọc frame đầu tiên để vẽ zone
            if st.session_state.first_frame is None and st.session_state.video_path:
                cap = cv2.VideoCapture(st.session_state.video_path)
                ret, frm = cap.read()
                cap.release()
                if ret:
                    st.session_state.first_frame = frm

        st.divider()
        st.markdown("### 🎯 Tham số")
        min_votes  = st.slider("Min votes để chốt",  1, 10, 3)
        ocr_intv   = st.slider("OCR interval (frames)", 1, 15, 5)
        skip       = st.slider("Skip frames", 1, 5, 2)

        st.divider()
        # Nút Start / Stop
        col_a, col_b = st.columns(2)
        with col_a:
            start_btn = st.button("▶ Bắt đầu", type="primary",
                                  disabled=st.session_state.running or st.session_state.video_path is None,
                                  use_container_width=True)
        with col_b:
            stop_btn  = st.button("⏹ Dừng",
                                  disabled=not st.session_state.running,
                                  use_container_width=True)

        if stop_btn:
            st.session_state.stop_event.set()
            st.session_state.running = False

        if start_btn and st.session_state.video_path:
            st.session_state.stop_event.clear()
            st.session_state.running     = True
            st.session_state.detections  = {}
            st.session_state.selected_id = None
            # Khởi động background thread
            fq = queue.Queue(maxsize=2)
            dq = queue.Queue(maxsize=200)
            st.session_state.frame_queue     = fq
            st.session_state.detection_queue = dq
            t = threading.Thread(
                target=pipeline_thread,
                args=(
                    st.session_state.video_path,
                    st.session_state.zone_polygon,
                    fq, dq,
                    st.session_state.stop_event,
                ),
                kwargs={
                    "min_votes":    min_votes,
                    "ocr_interval": ocr_intv,
                    "skip":         skip,
                },
                daemon=True,
            )
            t.start()
            st.session_state.thread = t

    # ── Layout chính: 2 cột ──────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="medium")

    # ── Cột TRÁI: Video + Zone Drawing ──────────────────────────────────────
    with col_left:

        if not st.session_state.running:
            # Chế độ vẽ Zone
            if st.session_state.first_frame is not None:
                st.markdown("#### 🎨 Vẽ Zone of Interest")
                st.markdown("""
                <div class="zone-info">
                    ✏️ <b>Click</b> lên khung hình để thêm điểm · <b>Double-click</b> để đóng polygon.<br>
                    Sau khi đóng, kéo các điểm để chỉnh lại vùng.<br>
                    Nếu không vẽ, hệ thống sẽ dùng <b>dải ngang mặc định (20%–85%)</b>.
                </div>
                """, unsafe_allow_html=True)

                zone = zone_canvas_component(
                    st.session_state.first_frame,
                    existing_points=st.session_state.zone_polygon,
                )
                if zone:
                    st.session_state.zone_polygon = zone
                    st.success(f"✅ Zone đã vẽ: {len(zone)} điểm")
                elif st.session_state.zone_polygon is None:
                    st.info("ℹ️ Chưa vẽ zone — sẽ dùng dải ngang mặc định")
            else:
                st.markdown("""
                <div class="video-placeholder">
                    📁 Upload video ở sidebar để bắt đầu
                </div>
                """, unsafe_allow_html=True)

        else:
            # Chế độ chạy pipeline — hiển thị video live
            st.markdown("#### 📹 Video Live")
            frame_placeholder = st.empty()
            info_placeholder  = st.empty()

            # Thu thập detection từ queue
            dq: queue.Queue = st.session_state.detection_queue
            while not dq.empty():
                try:
                    item = dq.get_nowait()
                    if "error" in item:
                        st.error(f"Lỗi pipeline: {item['error']}")
                        if "tb" in item:
                            st.code(item["tb"])
                        st.session_state.running = False
                        break
                    if "done" in item:
                        st.session_state.running = False
                        st.success("✅ Video đã xử lý xong!")
                        break
                    # Cập nhật detection
                    tid = item["track_id"]
                    existing = st.session_state.detections.get(tid)
                    # Ưu tiên cập nhật nếu mới hơn (đã chốt)
                    if existing is None or (item["confirmed"] and not existing.confirmed):
                        st.session_state.detections[tid] = DetectionRecord(
                            track_id    = tid,
                            plate_text  = item["plate_text"],
                            conf        = item["conf"],
                            confirmed   = item["confirmed"],
                            vehicle_img = item.get("vehicle_img"),
                            plate_img   = item.get("plate_img"),
                            class_name  = item.get("class_name", "vehicle"),
                        )
                except queue.Empty:
                    break

            # Hiển thị frame mới nhất
            fq: queue.Queue = st.session_state.frame_queue
            latest_frame = None
            while not fq.empty():
                try:
                    latest_frame = fq.get_nowait()
                except queue.Empty:
                    break

            if latest_frame:
                frame_placeholder.image(
                    latest_frame["frame"],
                    channels="RGB",
                    use_container_width=True,
                )
                fps  = latest_frame.get("fps", 0)
                fidx = latest_frame.get("idx", 0)
                info_placeholder.caption(f"Frame: {fidx} · FPS: {fps:.1f}")
            else:
                frame_placeholder.markdown("""
                <div class="video-placeholder" style="height:200px;">
                    ⏳ Đang khởi động pipeline...
                </div>
                """, unsafe_allow_html=True)

            # Auto-refresh khi đang chạy
            if st.session_state.running:
                time.sleep(0.08)
                st.rerun()

    # ── Cột PHẢI: Kết quả biển số ────────────────────────────────────────────
    with col_right:
        st.markdown("#### 📋 Kết quả nhận diện")

        detections: Dict[int, DetectionRecord] = st.session_state.detections

        # Stats
        n_confirmed = sum(1 for r in detections.values() if r.confirmed)
        n_pending   = len(detections) - n_confirmed

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">{len(detections)}</div>
                <div class="stat-label">Tổng xe</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value" style="color:#22c55e">{n_confirmed}</div>
                <div class="stat-label">Đã chốt</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value" style="color:#f59e0b">{n_pending}</div>
                <div class="stat-label">Chờ</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not detections:
            st.markdown("""
            <div style="text-align:center; color:#4a5568; padding:40px 20px;">
                <div style="font-size:2.5rem">🚗</div>
                <div style="margin-top:10px">Chưa phát hiện biển số nào.<br>
                <small>Bắt đầu nhận diện từ sidebar.</small></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Sắp xếp: đã chốt lên đầu
            sorted_recs = sorted(
                detections.values(),
                key=lambda r: (not r.confirmed, -r.conf),
            )

            for rec in sorted_recs:
                is_selected = (st.session_state.selected_id == rec.track_id)
                clicked = render_detection_card(rec, is_selected)
                if clicked:
                    if st.session_state.selected_id == rec.track_id:
                        st.session_state.selected_id = None  # Toggle off
                    else:
                        st.session_state.selected_id = rec.track_id

            # Hiển thị ảnh khi click
            if st.session_state.selected_id is not None:
                sel = detections.get(st.session_state.selected_id)
                if sel:
                    st.divider()
                    st.markdown(f"#### 🔍 Chi tiết: `{sel.plate_text}`")

                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.markdown("**Ảnh xe**")
                        if sel.vehicle_img is not None:
                            pil = numpy_to_pil(sel.vehicle_img)
                            if pil:
                                st.image(pil, use_container_width=True)
                        else:
                            st.caption("_(chưa có ảnh)_")

                    with img_col2:
                        st.markdown("**Ảnh biển số**")
                        if sel.plate_img is not None:
                            pil = numpy_to_pil(sel.plate_img)
                            if pil:
                                st.image(pil, use_container_width=True)
                        else:
                            st.caption("_(chưa có ảnh)_")

                    st.caption(
                        f"ID #{sel.track_id} · {sel.class_name} · "
                        f"conf {sel.conf*100:.0f}% · "
                        f"{'✓ Đã chốt' if sel.confirmed else '⏳ Chờ'}"
                    )


if __name__ == "__main__":
    main()
