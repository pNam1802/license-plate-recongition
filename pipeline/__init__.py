"""
pipeline/ — Các module xử lý chính của hệ thống nhận diện biển số xe.

Modules:
    vehicle_detector  — Stage 1+2: Phát hiện xe + ByteTrack tracking
    plate_reader      — Stage 3+4+5: Phát hiện biển số + Warp + PaddleOCR
    vote_logic        — Stage 6+7: Regex filter + Voting
"""
