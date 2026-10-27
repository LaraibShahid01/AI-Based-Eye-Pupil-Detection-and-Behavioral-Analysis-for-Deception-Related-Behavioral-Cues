"""FYP eye-video feasibility analysis.

Outputs image-space estimates, NOT clinical pupil measurements or deception detection.
MediaPipe iris landmarks are iris boundaries, not pupil boundaries. Pupil estimates
from dark-region segmentation are rejected when the image is ambiguous.
"""
import argparse
import csv
import math
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
# MediaPipe's left/right indices refer to image conventions, not anatomical left/right.
EYES = [
    {"name": "eye_33", "corners": (33, 133), "pairs": ((159, 145), (160, 144), (158, 153)),
     "iris": (468, 469, 470, 471, 472), "outline": (33, 160, 158, 133, 153, 144)},
    {"name": "eye_263", "corners": (362, 263), "pairs": ((386, 374), (385, 380), (387, 373)),
     "iris": (473, 474, 475, 476, 477), "outline": (362, 385, 387, 263, 373, 380)},
]


def pt(lms, i, w, h):
    return np.array((lms[i].x * w, lms[i].y * h), dtype=np.float32)


def norm(a, b):
    return float(np.linalg.norm(a - b))


def pupil_segment(gray, center, iris_radius, eye_polygon):
    """Dark-blob segmentation constrained to the iris and visible eye.

    Returns (center_xy, diameter_px, confidence) or None. The confidence is a
    heuristic quality score, NOT a calibrated probability.
    """
    H, W = gray.shape
    cx, cy = map(float, center)
    if iris_radius < 3.5:
        return None
    radius = max(5, int(round(iris_radius * 1.05)))
    x0, x1 = max(0, int(cx) - radius), min(W, int(cx) + radius + 1)
    y0, y1 = max(0, int(cy) - radius), min(H, int(cy) + radius + 1)
    if x1 - x0 < 9 or y1 - y0 < 9:
        return None
    roi = gray[y0:y1, x0:x1]
    smooth = cv2.GaussianBlur(roi, (3, 3), 0)
    eye_mask = np.zeros(roi.shape, dtype=np.uint8)
    poly = np.round(eye_polygon - np.array([x0, y0])).astype(np.int32)
    cv2.fillPoly(eye_mask, [poly], 255)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    iris_mask = (((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.86 * iris_radius) ** 2)
    mask = (eye_mask > 0) & iris_mask
    values = smooth[mask]
    if values.size < 20:
        return None
    # Avoid using an absolute darkness threshold: adapt to illumination.
    low, high = np.percentile(values, [15, 75])
    if high - low < 9:  # not enough contrast to distinguish a pupil
        return None
    threshold = float(min(np.percentile(values, 33), low + 0.52 * (high - low)))
    binary = np.zeros(roi.shape, dtype=np.uint8)
    binary[(smooth <= threshold) & mask] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    iris_area = math.pi * iris_radius ** 2
    for k in range(1, count):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < max(5, iris_area * 0.035) or area > iris_area * 0.67:
            continue
        px, py = centers[k]
        dist = math.hypot(px + x0 - cx, py + y0 - cy)
        if dist > 0.42 * iris_radius:
            continue
        component = (labels == k).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
        circularity = 4 * math.pi * area / max(perimeter * perimeter, 1)
        if circularity < 0.28:
            continue
        equivalent_diameter = 2 * math.sqrt(area / math.pi)
        if not (0.18 * iris_radius * 2 <= equivalent_diameter <= 0.83 * iris_radius * 2):
            continue
        score = float(np.clip(0.5 * circularity + 0.3 * (1 - dist / iris_radius) +
                              0.2 * min((high - low) / 55, 1), 0, 1))
        candidates.append((score, (px + x0, py + y0), equivalent_diameter))
    if not candidates:
        return None
    score, pos, diameter = max(candidates, key=lambda item: item[0])
    if score < 0.54:
        return None
    return np.array(pos, dtype=np.float32), float(diameter), score


def eye_measure(lms, eye, w, h, gray):
    a, b = [pt(lms, i, w, h) for i in eye["corners"]]
    width = norm(a, b)
    if width < 12:
        return None
    gaps = [norm(pt(lms, u, w, h), pt(lms, d, w, h)) for u, d in eye["pairs"]]
    ear = float(np.mean(gaps) / width)
    iris_center = pt(lms, eye["iris"][0], w, h)
    iris_boundary = [pt(lms, i, w, h) for i in eye["iris"][1:]]
    iris_radius = float(np.median([norm(iris_center, q) for q in iris_boundary]))
    eye_poly = np.array([pt(lms, i, w, h) for i in eye["outline"]])
    pupil = pupil_segment(gray, iris_center, iris_radius, eye_poly) if ear > 0.13 else None
    # Gaze is an uncalibrated iris-position proxy, not gaze in degrees.
    x_axis = (b - a) / width
    gaze_x = float(np.dot(iris_center - a, x_axis) / width)
    upper = np.mean([pt(lms, pair[0], w, h) for pair in eye["pairs"]], axis=0)
    lower = np.mean([pt(lms, pair[1], w, h) for pair in eye["pairs"]], axis=0)
    v = lower - upper
    gaze_y = float(np.dot(iris_center - upper, v) / max(np.dot(v, v), 1))
    return {"ear": ear, "iris_center": iris_center, "iris_radius": iris_radius,
            "iris_boundary": iris_boundary, "eye_polygon": eye_poly, "pupil": pupil,
            "gaze": np.array([gaze_x, gaze_y], dtype=np.float32)}


def draw_eye(frame, m):
    cv2.polylines(frame, [np.round(m["eye_polygon"]).astype(np.int32)], True, (0, 220, 0), 1)
    cv2.circle(frame, tuple(np.round(m["iris_center"]).astype(int)), 3, (0, 255, 255), -1)
    for q in m["iris_boundary"]:
        cv2.circle(frame, tuple(np.round(q).astype(int)), 2, (0, 255, 255), -1)
    if m["pupil"] is not None:
        c, d, confidence = m["pupil"]
        cv2.circle(frame, tuple(np.round(c).astype(int)), max(1, round(d / 2)), (255, 0, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="FYP eye behavior camera feasibility test")
    ap.add_argument("--video", required=True, help="Input video e.g. C7968.MP4")
    ap.add_argument("--output", default="results", help="Output directory")
    ap.add_argument("--model", default="face_landmarker.task")
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--blink-close", type=float, default=0.17,
                    help="EAR below this starts a blink; calibrate with manual labels")
    ap.add_argument("--blink-open", type=float, default=0.205,
                    help="EAR above this ends a blink; must exceed --blink-close")
    ap.add_argument("--baseline-seconds", type=float, default=3.0,
                    help="Initial recording interval used as neutral reference; only valid if truly neutral")
    ap.add_argument("--fixation-window", type=float, default=0.20)
    ap.add_argument("--fixation-dispersion", type=float, default=0.075,
                    help="Normalized gaze dispersion threshold, uncalibrated")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    if args.blink_open <= args.blink_close:
        ap.error("--blink-open must be greater than --blink-close")
    if args.max_width < 320:
        ap.error("--max-width must be >= 320")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = Path(args.model)
    if not model.is_file():
        print("Downloading MediaPipe Face Landmarker model...", flush=True)
        urlretrieve(MODEL_URL, model)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model.resolve())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps < 1:
        fps = 30.0
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_w <= 0 or original_h <= 0:
        raise RuntimeError("Could not read video dimensions")
    scale = min(1.0, args.max_width / original_w)
    w, h = max(2, round(original_w * scale)), max(2, round(original_h * scale))
    writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Could not create annotated.mp4")

    fields = ["frame", "time_s", "face_detected", "eyes_detected", "ear", "eye_open",
              "blink_count", "blink_frequency_per_min", "last_blink_duration_ms",
              "iris_diameter_px_proxy", "pupil_diameter_px_est", "pupil_iris_ratio_est",
              "pupil_quality", "pupil_change_pct_from_baseline", "pupil_dilation_pct_from_baseline",
              "pupil_constriction_pct_from_baseline", "relative_gaze_x", "relative_gaze_y",
              "gaze_deviation_from_baseline", "eye_movement_speed_relative_per_s",
              "fixation_candidate_s"]
    rows = []
    blink_count = 0
    blink_start = None
    blink_durations = []
    last_blink_duration = float("nan")
    last_gaze = None
    last_gaze_time = None
    gaze_history = deque()
    fixation_start = None
    baseline_pupils = []
    baseline_gazes = []
    baseline_pupil = None
    baseline_gaze = None
    n_eyes = n_pupil = idx = 0
    stop_requested = False

    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if scale != 1:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                t = idx / fps
                # MediaPipe VIDEO mode needs strictly increasing timestamps.
                timestamp_ms = int(round(t * 1000))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                result = detector.detect_for_video(image, timestamp_ms)
                values = {name: float("nan") for name in fields[4:]}
                values.update(blink_count=blink_count, eye_open="")
                face = bool(result.face_landmarks)
                eyes_ok = False
                if face and len(result.face_landmarks[0]) >= 478:
                    lms = result.face_landmarks[0]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    measurements = [eye_measure(lms, eye, w, h, gray) for eye in EYES]
                    eyes_ok = all(m is not None for m in measurements)
                    if eyes_ok:
                        n_eyes += 1
                        ear = float(np.mean([m["ear"] for m in measurements]))
                        is_closed = ear < (args.blink_open if blink_start is not None else args.blink_close)
                        if is_closed and blink_start is None:
                            blink_start = t
                        elif not is_closed and blink_start is not None:
                            duration = t - blink_start
                            if 0.05 <= duration <= 0.8:
                                blink_count += 1
                                last_blink_duration = duration * 1000
                                blink_durations.append(last_blink_duration)
                            blink_start = None
                        gaze = np.mean([m["gaze"] for m in measurements], axis=0)
                        pupils = [m["pupil"] for m in measurements if m["pupil"] is not None]
                        # For a valid pupil estimate require both eyes and a reasonable size agreement.
                        pupil_ratio = pupil_diameter = pupil_quality = float("nan")
                        if len(pupils) == 2:
                            ratios = [m["pupil"][1] / (2 * m["iris_radius"]) for m in measurements]
                            if abs(ratios[0] - ratios[1]) <= 0.25:
                                pupil_ratio = float(np.mean(ratios))
                                pupil_diameter = float(np.mean([p[1] for p in pupils]))
                                pupil_quality = float(np.mean([p[2] for p in pupils]))
                                n_pupil += 1
                        iris_diameter = float(np.mean([2 * m["iris_radius"] for m in measurements]))
                        if t < args.baseline_seconds:
                            baseline_gazes.append(gaze.copy())
                            if math.isfinite(pupil_ratio):
                                baseline_pupils.append(pupil_ratio)
                        if t >= args.baseline_seconds and baseline_gaze is None and baseline_gazes:
                            baseline_gaze = np.median(np.array(baseline_gazes), axis=0)
                        if t >= args.baseline_seconds and baseline_pupil is None and len(baseline_pupils) >= 5:
                            baseline_pupil = float(np.median(baseline_pupils))
                        pupil_change = float("nan")
                        if baseline_pupil is not None and math.isfinite(pupil_ratio):
                            pupil_change = 100 * (pupil_ratio / baseline_pupil - 1)
                        deviation = float(np.linalg.norm(gaze - baseline_gaze)) if baseline_gaze is not None else float("nan")
                        speed = float("nan")
                        if last_gaze is not None:
                            speed = float(np.linalg.norm(gaze - last_gaze) / max(t - last_gaze_time, 1 / fps))
                        last_gaze, last_gaze_time = gaze.copy(), t
                        if not is_closed:
                            gaze_history.append((t, gaze.copy()))
                        else:
                            gaze_history.clear()
                            fixation_start = None
                        while gaze_history and t - gaze_history[0][0] > args.fixation_window:
                            gaze_history.popleft()
                        fixation = 0.0
                        if len(gaze_history) >= 3 and gaze_history[-1][0] - gaze_history[0][0] >= args.fixation_window * 0.8:
                            samples = np.array([g for _, g in gaze_history])
                            dispersion = float(np.ptp(samples[:, 0]) + np.ptp(samples[:, 1]))
                            if dispersion < args.fixation_dispersion:
                                if fixation_start is None:
                                    fixation_start = gaze_history[0][0]
                                fixation = t - fixation_start
                            else:
                                fixation_start = None
                        values.update(ear=ear, eye_open=int(not is_closed), blink_count=blink_count,
                                      blink_frequency_per_min=blink_count * 60 / max(t, 1 / fps),
                                      last_blink_duration_ms=last_blink_duration,
                                      iris_diameter_px_proxy=iris_diameter,
                                      pupil_diameter_px_est=pupil_diameter,
                                      pupil_iris_ratio_est=pupil_ratio, pupil_quality=pupil_quality,
                                      pupil_change_pct_from_baseline=pupil_change,
                                      pupil_dilation_pct_from_baseline=max(pupil_change, 0) if math.isfinite(pupil_change) else float("nan"),
                                      pupil_constriction_pct_from_baseline=max(-pupil_change, 0) if math.isfinite(pupil_change) else float("nan"),
                                      relative_gaze_x=float(gaze[0]), relative_gaze_y=float(gaze[1]),
                                      gaze_deviation_from_baseline=deviation,
                                      eye_movement_speed_relative_per_s=speed,
                                      fixation_candidate_s=fixation)
                        for m in measurements:
                            draw_eye(frame, m)
                        label = f"EAR {ear:.3f} | blinks {blink_count} | pupil/iris {pupil_ratio:.3f}" if math.isfinite(pupil_ratio) else f"EAR {ear:.3f} | blinks {blink_count} | pupil unavailable"
                        cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                if not eyes_ok:
                    # Never count missing face frames as blinks.
                    blink_start = None
                    last_gaze = last_gaze_time = fixation_start = None
                    gaze_history.clear()
                    cv2.putText(frame, "Eye landmarks unavailable", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                rows.append(dict(frame=idx, time_s=round(t, 4), face_detected=int(face),
                                 eyes_detected=int(eyes_ok), **values))
                writer.write(frame)
                idx += 1
                if idx % 120 == 0:
                    print(f"Processed {idx} frames / {t:.1f}s", flush=True)
                if args.preview:
                    cv2.imshow("FYP Eye Analysis (press Q to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_requested = True
                        break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    with (output / "features.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    duration = idx / fps
    mean_blink = f"{np.mean(blink_durations):.1f} ms" if blink_durations else "unavailable"
    report = (f"FYP CAMERA FEASIBILITY REPORT\n"
              f"Input: {args.video}\nOriginal: {original_w}x{original_h} at {fps:.2f} FPS\n"
              f"Processed: {w}x{h}; {idx} frames; {duration:.2f}s\n"
              f"Eye-landmark success: {n_eyes}/{idx} ({100 * n_eyes / max(idx, 1):.1f}%)\n"
              f"Two-eye pupil-segmentation acceptance: {n_pupil}/{idx} ({100 * n_pupil / max(idx, 1):.1f}%)\n"
              f"Blinks (heuristic): {blink_count}\nBlink rate: {60 * blink_count / max(duration, 1e-9):.2f}/min\n"
              f"Mean blink duration: {mean_blink}\n"
              f"Pupil baseline: {'available' if baseline_pupil is not None else 'unavailable'}\n"
              f"Gaze baseline: {'available' if baseline_gaze is not None else 'unavailable'}\n"
              f"Stopped early by user: {stop_requested}\n\n"
              "LIMITATIONS:\n"
              "Pupil segmentation is experimental and may mistake dark iris, eyelashes, reflections or shadows for pupil.\n"
              "Do not use pupil results unless visually verified against labeled frames.\n"
              "Pupil size is in pixels and pupil/iris ratios, NOT millimeters.\n"
              "First baseline interval must actually contain neutral behavior; otherwise baseline is invalid.\n"
              "Gaze, speed and fixation are uncalibrated image-space proxies, NOT gaze degrees or validated fixations.\n"
              "Blink thresholds must be calibrated per subject and manually validated.\n"
              "No deception detection or lie classification is performed.\n")
    (output / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
