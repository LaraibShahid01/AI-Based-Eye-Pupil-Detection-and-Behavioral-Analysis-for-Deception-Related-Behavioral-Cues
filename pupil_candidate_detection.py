
from pathlib import Path
import csv
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
INPUT = BASE / "results" / "day4" / "original_quality"
OUTPUT = BASE / "results" / "day4" / "pupil_candidates"
OUTPUT.mkdir(parents=True, exist_ok=True)

images = sorted(INPUT.glob("*_original.png"))

if not images:
    raise RuntimeError(
        "No original eye crops found. "
        "Run: python original_eye_quality.py"
    )

rows = []

for path in images:
    img = cv2.imread(str(path))

    if img is None:
        print("Cannot read:", path.name)
        continue

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # Smooth sensor noise
    blurred = cv2.GaussianBlur(
        enhanced, (5, 5), 0
    )

    # Threshold darker regions using Otsu
    threshold_value, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Limit analysis to a central ellipse.
    # Experimental ROI, NOT an iris mask.
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        roi,
        (w // 2, h // 2),
        (
            max(1, int(w * 0.25)),
            max(1, int(h * 0.25))
        ),
        0, 0, 360, 255, -1
    )

    binary = cv2.bitwise_and(
        binary, binary, mask=roi
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    roi_area = float(np.count_nonzero(roi))

    for contour in contours:
        area = cv2.contourArea(contour)

        if area <= 0 or roi_area <= 0:
            continue

        area_ratio = area / roi_area

        if not (0.005 <= area_ratio <= 0.40):
            continue

        perimeter = cv2.arcLength(
            contour, True
        )

        if perimeter <= 0:
            continue

        circularity = (
            4 * np.pi * area / (perimeter ** 2)
        )

        (cx, cy), radius = cv2.minEnclosingCircle(
            contour
        )

        # Distance from crop center.
        # This assumes approximate frontal gaze.
        dx = (cx - w / 2) / max(w / 2, 1)
        dy = (cy - h / 2) / max(h / 2, 1)

        center_distance = np.sqrt(
            dx * dx + dy * dy
        )

        if circularity < 0.25:
            continue

        # A screening score, NOT probability.
        score = (
            0.7 * min(circularity, 1.0)
            + 0.3 * max(0, 1 - center_distance)
        )

        candidates.append({
            "contour": contour,
            "area": area,
            "circularity": circularity,
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "score": score
        })

    output_img = img.copy()

    # Show experimental search region
    cv2.ellipse(
        output_img,
        (w // 2, h // 2),
        (
            max(1, int(w * 0.25)),
            max(1, int(h * 0.25))
        ),
        0, 0, 360, (255, 255, 0), 1
    )

    status = "no_candidate"
    best = None

    if candidates:
        best = max(
            candidates,
            key=lambda c: c["score"]
        )
        status = "candidate_found"

        cv2.drawContours(
            output_img,
            [best["contour"]],
            -1,
            (0, 255, 0),
            2
        )

        cv2.circle(
            output_img,
            (
                round(best["cx"]),
                round(best["cy"])
            ),
            2,
            (0, 0, 255),
            -1
        )

    name = path.stem.replace(
        "_original", ""
    )

    # Save original-scale evidence
    cv2.imwrite(
        str(OUTPUT / f"{name}_candidate.png"),
        output_img
    )

    cv2.imwrite(
        str(OUTPUT / f"{name}_mask.png"),
        binary
    )

    rows.append({
        "filename": path.name,
        "status": status,
        "threshold": round(threshold_value, 2),
        "candidate_area_px2": (
            round(best["area"], 2)
            if best else ""
        ),
        "circularity": (
            round(best["circularity"], 3)
            if best else ""
        ),
        "center_x_px": (
            round(best["cx"], 2)
            if best else ""
        ),
        "center_y_px": (
            round(best["cy"], 2)
            if best else ""
        ),
        "radius_px": (
            round(best["radius"], 2)
            if best else ""
        ),
        "screening_score": (
            round(best["score"], 3)
            if best else ""
        )
    })

    print(path.name, "->", status)

csv_path = OUTPUT / "candidate_results.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

found = sum(
    x["status"] == "candidate_found"
    for x in rows
)

report = f"""EXPERIMENTAL PUPIL CANDIDATE REPORT

Original eye crops analyzed: {len(rows)}
Images with a dark-region candidate: {found}
Images without accepted candidate: {len(rows) - found}

Method:
CLAHE, Gaussian smoothing, Otsu threshold,
central ROI, contour screening.

These are dark-region candidates only.
Candidate found does NOT mean true pupil.
Screening score is NOT accuracy or confidence.
No validated pupil diameter is reported.

Next:
Inspect candidate overlays against
manually marked pupil boundaries.
"""

(OUTPUT / "candidate_report.txt").write_text(
    report, encoding="utf-8"
)

print("\n" + report)
print("Saved to:", OUTPUT)
