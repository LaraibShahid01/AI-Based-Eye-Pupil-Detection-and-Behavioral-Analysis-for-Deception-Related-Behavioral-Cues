
from pathlib import Path
import csv
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from eye_visualization import EYES, get_eye_crop

BASE = Path(__file__).resolve().parent
VIDEO = BASE / "C7968.MP4"
MODEL = BASE / "face_landmarker.task"

OUT = BASE / "results" / "day4" / "original_quality"
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or total <= 0:
    cap.release()
    raise RuntimeError("Invalid video metadata")

indices = sorted({
    round(i * (total - 1) / 9)
    for i in range(10)
})

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=str(MODEL)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

rows = []

with vision.FaceLandmarker.create_from_options(
    options
) as detector:

    for frame_id in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if not ok:
            print("Unable to read frame", frame_id)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )

        timestamp_ms = round(frame_id / fps * 1000)

        result = detector.detect_for_video(
            image, timestamp_ms
        )

        if not result.face_landmarks:
            print("No face at frame", frame_id)
            continue

        landmarks = result.face_landmarks[0]

        if len(landmarks) < 478:
            continue

        for eye_name, eye in EYES.items():

            crop = get_eye_crop(
                frame, landmarks, eye, w, h
            )

            if crop.size == 0:
                continue

            gray = cv2.cvtColor(
                crop, cv2.COLOR_BGR2GRAY
            )

            original_sharpness = float(
                cv2.Laplacian(
                    gray, cv2.CV_64F
                ).var()
            )

            enlarged = cv2.resize(
                crop, None,
                fx=4, fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            zoom_gray = cv2.cvtColor(
                enlarged, cv2.COLOR_BGR2GRAY
            )

            zoom_sharpness = float(
                cv2.Laplacian(
                    zoom_gray, cv2.CV_64F
                ).var()
            )

            filename = (
                f"frame_{frame_id:06d}_{eye_name}"
            )

            cv2.imwrite(
                str(OUT / f"{filename}_original.png"),
                crop
            )

            rows.append({
                "frame": frame_id,
                "eye": eye_name,
                "crop_width": crop.shape[1],
                "crop_height": crop.shape[0],
                "original_sharpness": round(
                    original_sharpness, 3
                ),
                "zoom_sharpness": round(
                    zoom_sharpness, 3
                ),
                "brightness": round(
                    float(gray.mean()), 2
                ),
                "contrast": round(
                    float(gray.std()), 2
                )
            })

        print("Processed sample frame:", frame_id)

cap.release()

if not rows:
    raise RuntimeError("No valid eye crops generated")

with (OUT / "original_quality.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

original_scores = [
    r["original_sharpness"] for r in rows
]
zoom_scores = [
    r["zoom_sharpness"] for r in rows
]

report = f"""ORIGINAL EYE QUALITY COMPARISON

Eye crops analyzed: {len(rows)}
Mean original sharpness proxy:
{np.mean(original_scores):.3f}

Mean 4x enlarged sharpness proxy:
{np.mean(zoom_scores):.3f}

Original sharpness range:
{min(original_scores):.3f} to {max(original_scores):.3f}

All scores are relative image-space metrics.
They do not establish pupil visibility,
pupil accuracy or camera suitability.
"""

(OUT / "comparison_report.txt").write_text(
    report, encoding="utf-8"
)

print("\n" + report)
print("Saved to:", OUT)

