
from pathlib import Path
import csv
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Project paths
BASE = Path(__file__).resolve().parent
VIDEO = BASE / "C7968.MP4"
MODEL = BASE / "face_landmarker.task"

OUT = BASE / "results" / "day4"
SAMPLES = OUT / "screenshots"
CROPS = OUT / "eye_crops"

SAMPLES.mkdir(parents=True, exist_ok=True)
CROPS.mkdir(parents=True, exist_ok=True)

# MediaPipe eye landmark indices.
# Names are landmark-based, not anatomical left/right.
EYES = {
    "eye_33": {
        "outline": [33, 160, 158, 133, 153, 144],
        "iris": [468, 469, 470, 471, 472],
    },
    "eye_263": {
        "outline": [362, 385, 387, 263, 373, 380],
        "iris": [473, 474, 475, 476, 477],
    },
}


def pixel_point(landmarks, index, w, h):
    p = landmarks[index]
    return (
        int(round(p.x * w)),
        int(round(p.y * h))
    )


def get_eye_crop(frame, landmarks, eye, w, h):
    # Use eye outline and iris to define a crop.
    ids = eye["outline"] + eye["iris"]
    points = np.array([
        pixel_point(landmarks, i, w, h)
        for i in ids
    ])

    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)

    eye_width = max(1, x1 - x0)
    eye_height = max(1, y1 - y0)

    pad_x = max(15, int(eye_width * 0.4))
    pad_y = max(15, int(eye_height * 0.8))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    return frame[y0:y1, x0:x1].copy()


def draw_landmarks(frame, landmarks, w, h):
    # Draw a sparse face-landmark visualization.
    # The model still returns all available landmarks.
    for i in range(0, min(468, len(landmarks)), 3):
        p = pixel_point(landmarks, i, w, h)
        cv2.circle(frame, p, 1, (255, 160, 0), -1)

    # Draw eye outlines and iris landmarks.
    for name, eye in EYES.items():
        pts = np.array([
            pixel_point(landmarks, i, w, h)
            for i in eye["outline"]
        ], dtype=np.int32)

        cv2.polylines(
            frame, [pts], True, (0, 255, 0), 2
        )

        for i in eye["iris"]:
            p = pixel_point(landmarks, i, w, h)
            cv2.circle(frame, p, 3, (0, 255, 255), -1)

        center = pixel_point(
            landmarks, eye["iris"][0], w, h
        )

        cv2.putText(
            frame, name,
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )


def main():
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)

    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Unable to open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata")

    print("FPS:", fps)
    print("Resolution:", w, "x", h)
    print("Reported frames:", total)

    # Select 10 representative frames across video.
    count = min(10, total) if total > 0 else 0
    sample_indices = set()

    if count > 0:
        sample_indices = {
            round(i * (total - 1) / max(count - 1, 1))
            for i in range(count)
        }

    video_path = OUT / "face_eye_iris_annotated.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    processed = 0
    detected = 0
    samples_saved = 0
    frame_rows = []

    try:
        with vision.FaceLandmarker.create_from_options(
            options
        ) as detector:

            while True:
                ok, original = cap.read()

                if not ok:
                    break

                frame = original.copy()

                # Estimated timestamp based on nominal FPS.
                time_s = processed / fps
                time_ms = round(time_s * 1000)

                rgb = cv2.cvtColor(
                    original, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )

                result = detector.detect_for_video(
                    mp_image, time_ms
                )

                valid = (
                    bool(result.face_landmarks)
                    and len(result.face_landmarks[0]) >= 478
                )

                if valid:
                    detected += 1
                    landmarks = result.face_landmarks[0]

                    draw_landmarks(
                        frame, landmarks, w, h
                    )

                else:
                    cv2.putText(
                        frame,
                        "Landmarks unavailable",
                        (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2
                    )

                frame_rows.append([
                    processed,
                    round(time_s, 6),
                    int(valid)
                ])

                # Save the annotated video frame.
                writer.write(frame)

                # Save original and annotated samples.
                if processed in sample_indices:
                    prefix = f"frame_{processed:06d}"

                    cv2.imwrite(
                        str(SAMPLES / f"{prefix}_original.jpg"),
                        original
                    )

                    cv2.imwrite(
                        str(SAMPLES / f"{prefix}_annotated.jpg"),
                        frame
                    )

                    if valid:
                        for name, eye in EYES.items():
                            crop = get_eye_crop(
                                original,
                                landmarks,
                                eye, w, h
                            )

                            if crop.size == 0:
                                continue

                            # Enlarge with interpolation.
                            zoom = cv2.resize(
                                crop, None,
                                fx=4, fy=4,
                                interpolation=cv2.INTER_CUBIC
                            )

                            cv2.imwrite(
                                str(
                                    CROPS /
                                    f"{prefix}_{name}_zoom.png"
                                ),
                                zoom
                            )

                    samples_saved += 1

                processed += 1

                if processed % 120 == 0:
                    print(
                        f"Processed {processed} frames"
                    )

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # Write frame detection log.
    with (OUT / "detection_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame",
            "time_seconds_est",
            "face_eye_landmarks_available"
        ])
        csv_writer.writerows(frame_rows)

    percentage = (
        100 * detected / processed
        if processed else 0
    )

    report = f"""FYP DAY 3 AND 4 VISUAL REPORT

Video: {VIDEO.name}
Resolution: {w} x {h}
FPS: {fps:.3f}
Decoded frames: {processed}
Estimated duration: {processed / fps:.3f}s

Frames with face/iris landmarks:
{detected}/{processed} ({percentage:.1f}%)

Representative frame positions: {samples_saved}
Annotation: Face dots, eye outlines,
and iris landmark points.

IMPORTANT:
Iris landmarks are NOT pupil boundaries.
Zooming an eye image adds no new detail.
Manual pupil-boundary assessment is required.
This is not a deception detector.
"""

    (OUT / "visual_report.txt").write_text(
        report, encoding="utf-8"
    )

    print("\n" + report)
    print("Output:", OUT)


if __name__ == "__main__":
    main()
