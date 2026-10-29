
from pathlib import Path
import csv
from collections import Counter

BASE = Path(__file__).resolve().parent
REVIEW = BASE / "results" / "day4" / "pupil_review"

CSV_FILE = REVIEW / "manual_review.csv"
REPORT = REVIEW / "validation_report.txt"

with CSV_FILE.open(
    "r", newline="", encoding="utf-8"
) as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise RuntimeError("No review records found")

visibility = Counter(
    r["manual_pupil_visible"].strip().lower()
    for r in rows
)

matches = Counter(
    r["candidate_matches_pupil"].strip().lower()
    for r in rows
)

pending = sum(
    r["manual_pupil_visible"].strip().lower()
    == "pending"
    or r["candidate_matches_pupil"].strip().lower()
    == "pending"
    for r in rows
)

text = f"""FYP PUPIL CANDIDATE VALIDATION

Total eye crops: {len(rows)}
Manual reviews incomplete: {pending}

Pupil visibility:
{dict(visibility)}

Candidate/pupil matches:
{dict(matches)}

CONCLUSION:
"""

if pending:
    text += (
        "Validation is still pending.\n"
        "Do not report candidate detection accuracy.\n"
    )
else:
    text += (
        "Manual review fields are complete.\n"
        "Further evaluation of annotation quality\n"
        "and ground truth is required.\n"
    )

REPORT.write_text(
    text, encoding="utf-8"
)

print(text)
print("Saved:", REPORT)
