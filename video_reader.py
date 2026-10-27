
from pathlib import Path
import csv
import cv2

# Locate the input video
BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "C7968.MP4"

# Create output folder
OUTPUT_DIR = BASE_DIR / "results" / "day2"
FRAMES_DIR = OUTPUT_DIR / "sample_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Open video
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Video could not be opened!")

# Read video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS")

print("Video opened successfully!")
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Reported frames:", total_frames)

# Choose 10 frames across the video
sample_indices = set()

if total_frames > 0:
    sample_count = min(10, total_frames)
    for i in range(sample_count):
        index = round(
            i * (total_frames - 1) /
            max(sample_count - 1, 1)
        )
        sample_indices.add(index)

# Read every frame and save timestamps
frame_count = 0
saved_count = 0

csv_path = OUTPUT_DIR / "frame_timestamps.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame_number",
        "time_seconds_est",
        "timestamp_ms_est",
        "timestamp_ns_est"
    ])

    while True:
        success, frame = cap.read()

        if not success:
            break

        time_s = frame_count / fps
        time_ms = round(time_s * 1000)
        time_ns = round(time_s * 1_000_000_000)

        writer.writerow([
            frame_count, round(time_s, 6),
            time_ms, time_ns
        ])

        if frame_count in sample_indices:
            filename = f"frame_{frame_count:06d}.jpg"
            path = FRAMES_DIR / filename

            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Failed to save {path}")

            saved_count += 1
            print("Saved:", filename)

        frame_count += 1

cap.release()

# Create video report
duration = frame_count / fps

report = f"""FYP DAY 2 VIDEO REPORT

Video: {VIDEO_PATH.name}
FPS: {fps:.3f}
Resolution: {width} x {height}
Reported frames: {total_frames}
Decoded frames: {frame_count}
Approximate duration: {duration:.3f} seconds
Frame interval: {1000/fps:.3f} milliseconds
Frame interval in ns: {1_000_000_000/fps:.0f}
Sample frames saved: {saved_count}

Timestamps are estimates based on frame index/FPS.
They are not hardware capture timestamps.
"""

report_path = OUTPUT_DIR / "video_info.txt"
report_path.write_text(report, encoding="utf-8")

print(report)
print("Day 2 processing finished!")
