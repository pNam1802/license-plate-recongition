"""
cut_video.py — Cắt video thành các đoạn ngắn bằng nhau
========================================================
Cách dùng:
    python tools/cut_video.py --source video.mp4 --duration 30
    python tools/cut_video.py --source video.mp4 --duration 10 --output-dir data/clips
    python tools/cut_video.py --source video.mp4 --duration 20 --start 60 --end 300

Tham số:
    --source / -s      File video đầu vào (bắt buộc)
    --duration / -d    Độ dài mỗi đoạn (giây, mặc định: 30)
    --output-dir / -o  Thư mục lưu kết quả (mặc định: cùng thư mục với video)
    --start            Bắt đầu cắt từ giây thứ mấy (mặc định: 0)
    --end              Kết thúc ở giây thứ mấy (mặc định: hết video)
    --prefix           Tiền tố tên file (mặc định: tên video gốc)
"""

import argparse
import os
import subprocess
import sys

import cv2


def _find_ffmpeg():
    """Tìm ffmpeg trong PATH."""
    for candidate in ("ffmpeg", "ffmpeg.exe"):
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def _cut_ffmpeg(ffmpeg, source, output_dir, prefix, segments):
    """Cắt bằng FFmpeg — không re-encode, rất nhanh."""
    saved = []
    total = len(segments)
    pad = len(str(total))

    for i, (t_start, t_end) in enumerate(segments, start=1):
        duration = t_end - t_start
        out_path = os.path.join(output_dir, f"{prefix}_{i:0{pad}d}.mp4")

        cmd = [
            ffmpeg, "-y",
            "-ss", str(t_start),
            "-i", source,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            out_path,
        ]

        print(f"  [{i}/{total}] {t_start:.1f}s → {t_end:.1f}s  ({duration:.1f}s)  →  {out_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [LỖI FFmpeg] {result.stderr[-300:]}")
        else:
            saved.append(out_path)

    return saved


def _cut_opencv(source, output_dir, prefix, segments):
    """Cắt bằng OpenCV — fallback khi không có FFmpeg."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {source}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    total = len(segments)
    pad   = len(str(total))
    saved = []

    for i, (t_start, t_end) in enumerate(segments, start=1):
        out_path = os.path.join(output_dir, f"{prefix}_{i:0{pad}d}.mp4")
        duration = t_end - t_start
        print(f"  [{i}/{total}] {t_start:.1f}s → {t_end:.1f}s  ({duration:.1f}s)  →  {out_path}")

        frame_start = int(t_start * fps)
        frame_end   = int(t_end   * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for _ in range(frame_end - frame_start):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        saved.append(out_path)

    cap.release()
    return saved


def _video_duration(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {source}")
        sys.exit(1)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps


def _build_segments(total_duration, clip_duration, start, end):
    end = min(end, total_duration)
    if start >= end:
        print(f"[LỖI] --start ({start}s) phải nhỏ hơn --end ({end}s).")
        sys.exit(1)

    segments = []
    t = start
    while t < end:
        t_end = min(t + clip_duration, end)
        segments.append((t, t_end))
        t = t_end
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Cắt video thành các đoạn ngắn bằng nhau",
    )
    parser.add_argument("--source", "-s", required=True, help="File video đầu vào")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="Độ dài mỗi đoạn (giây, mặc định: 30)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Thư mục lưu kết quả (mặc định: cùng thư mục video)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Bắt đầu cắt từ giây thứ mấy (mặc định: 0)")
    parser.add_argument("--end", type=float, default=None,
                        help="Kết thúc ở giây thứ mấy (mặc định: hết video)")
    parser.add_argument("--prefix", default=None,
                        help="Tiền tố tên file clip (mặc định: tên video gốc)")
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"[LỖI] Không tìm thấy file: {args.source}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.source))
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.source))[0]
    prefix    = args.prefix or base_name

    total_duration = _video_duration(args.source)
    end = args.end if args.end is not None else total_duration

    print(f"\n[INFO] Video:      {args.source}")
    print(f"[INFO] Thời lượng: {total_duration:.1f}s  ({total_duration/60:.1f} phút)")
    print(f"[INFO] Cắt từ:     {args.start}s → {end}s")
    print(f"[INFO] Mỗi đoạn:   {args.duration}s")
    print(f"[INFO] Lưu vào:    {output_dir}/\n")

    segments = _build_segments(total_duration, args.duration, args.start, end)
    print(f"[INFO] Sẽ tạo {len(segments)} đoạn:\n")

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        print(f"[INFO] Dùng FFmpeg ({ffmpeg}) — không re-encode, rất nhanh\n")
        saved = _cut_ffmpeg(ffmpeg, args.source, output_dir, prefix, segments)
    else:
        print("[INFO] Không tìm thấy FFmpeg — dùng OpenCV (chậm hơn)\n")
        saved = _cut_opencv(args.source, output_dir, prefix, segments)

    print(f"\n[XONG] Đã lưu {len(saved)}/{len(segments)} đoạn vào: {output_dir}/")


if __name__ == "__main__":
    main()
