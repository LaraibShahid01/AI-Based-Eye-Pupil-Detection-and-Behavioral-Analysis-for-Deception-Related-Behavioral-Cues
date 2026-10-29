
from pathlib import Path
import csv
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "results" / "day4" / "eye_crops"
OUTPUT_DIR = BASE / "results" / "day4" / "quality_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*_zoom.png"))

if not files:
    raise RuntimeError("No eye crop images found!")

results = []

for path in files:
    image = cv2.imread(str(path))

    if image is None:
        print("Could not read:", path.name)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur/sharpness proxy
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    # Average lighting
    brightness = float(np.mean(gray))

    # Overall intensity contrast
    contrast = float(np.std(gray))

    # Very bright pixel proportion
    # This is a highlight proxy, not confirmed
    # specular reflections.
    bright_fraction = float(
        np.mean(gray >= 245) * 100
    )

    # Dark-pixel ratio
    # This cannot distinguish pupil from dark iris,
    # eyelashes or shadows.
    dark_fraction = float(
        np.mean(gray <= 60) * 100
    )

    # Experimental screening thresholds only.
    # These require calibration against real images.
    flags = []

    if sharpness < 25:
        flags.append("possibly_blurry")

    if brightness < 45:
        flags.append("possibly_dark")

    if brightness > 220:
        flags.append("possibly_overexposed")

    if contrast < 25:
        flags.append("low_contrast")

    if bright_fraction > 10:
        flags.append("many_bright_pixels")

    status = "review" if flags else "no_global_flags"

    results.append({
        "filename": path.name,
        "sharpness_proxy": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "bright_pixels_pct": round(bright_fraction, 2),
        "dark_pixels_pct": round(dark_fraction, 2),
        "screening_status": status,
        "flags": ";".join(flags)
    })

csv_path = OUTPUT_DIR / "quality_results.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

report = [
    "FYP AUTOMATED EYE IMAGE QUALITY REPORT",
    "",
    f"Images analyzed: {len(results)}",
    f"Images flagged for review: "
    f"{sum(r['screening_status'] == 'review' for r in results)}",
    "",
    "All quality thresholds are experimental.",
    "Crops were enlarged 4x and interpolation",
    "may affect sharpness measurements.",
    "Dark pixels are NOT confirmed pupils.",
    "No pupil detection accuracy is measured."
]

report_path = OUTPUT_DIR / "quality_report.txt"
report_path.write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n".join(report))
print("Saved:", csv_path)

