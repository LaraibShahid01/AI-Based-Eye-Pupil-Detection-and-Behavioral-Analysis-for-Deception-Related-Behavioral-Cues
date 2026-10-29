
from pathlib import Path
import csv
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent

ORIGINAL = BASE / "results/day4/original_quality"
CANDIDATES = BASE / "results/day4/pupil_candidates"
OUT = BASE / "results/day4/pupil_review"

OUT.mkdir(parents=True, exist_ok=True)

csv_file = CANDIDATES / "candidate_results.csv"

with csv_file.open(
    "r", newline="", encoding="utf-8"
) as f:
    records = list(csv.DictReader(f))

panels = []
review_rows = []

for record in records:
    filename = record["filename"]
    name = Path(filename).stem.replace(
        "_original", ""
    )

    original = cv2.imread(
        str(ORIGINAL / filename)
    )

    candidate = cv2.imread(
        str(CANDIDATES / f"{name}_candidate.png")
    )

    mask = cv2.imread(
        str(CANDIDATES / f"{name}_mask.png")
    )

    if any(x is None for x in (
        original, candidate, mask
    )):
        print("Missing image:", filename)
        continue

    # Resize for display only
    target = (300, 180)

    original = cv2.resize(original, target)
    candidate = cv2.resize(candidate, target)
    mask = cv2.resize(mask, target)

    panel = np.hstack([
        original, candidate, mask
    ])

    # Add heading
    heading = np.full(
        (52, panel.shape[1], 3),
        245, dtype=np.uint8
    )

    cv2.putText(
        heading,
        f"{name} | {record['status']}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 0), 2
    )

    panel = np.vstack([heading, panel])

    panels.append(panel)

    review_rows.append({
        "filename": filename,
        "algorithm_status": record["status"],
        "manual_pupil_visible": "pending",
        "candidate_matches_pupil": "pending",
        "notes": ""
    })

# Create sheets of five images each
for start in range(0, len(panels), 5):
    sheet = np.vstack(
        panels[start:start + 5]
    )

    output_name = (
        f"review_sheet_{start // 5 + 1}.png"
    )

    cv2.imwrite(
        str(OUT / output_name),
        sheet
    )

if review_rows:
    with (OUT / "manual_review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(review_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(review_rows)

print("Review sheets generated:", len(panels))
print("Saved in:", OUT)
