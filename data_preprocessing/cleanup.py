"""
Removes job postings whose job_title appears fewer than --min-count times,
and removes postings whose location is not in the permitted Swiss cantons list.
Writes a new JSONL file — the original is never modified.

Usage:
    python data_preprocessing/cleanup.py
    python data_preprocessing/cleanup.py --input data/structured_jobs_20.05_normalized.jsonl
    python data_preprocessing/cleanup.py --input data/structured_jobs_20.05_normalized.jsonl --output data/cleaned.jsonl --min-count 20
"""
import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("data/structured_jobs_20.05_normalized.jsonl")
DEFAULT_MIN_COUNT = 20

VALID_LOCATIONS = {
    "Zurich", "Bern", "Aargau", "Lucerne", "Multiple Cantons",
    "Vaud", "St. Gallen", "Basel-Stadt", "Zug", "Solothurn",
    "Geneva", "Others", "Basel-Landschaft", "Thurgau", "Graubünden",
    "Valais", "Fribourg", "Neuchâtel", "Schaffhausen", "Schwyz",
    "Uri", "Jura", "Obwalden", "Glarus", "Nidwalden",
    "Appenzell Ausserrhoden", "Appenzell Innerrhoden", "Not Specified",
}


def fix_mojibake(s: str) -> str:
    """Fix UTF-8 text that was incorrectly decoded as Latin-1 (e.g. 'Ã¼' -> 'ü')."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def cleanup(input_path: Path, output_path: Path, min_count: int = DEFAULT_MIN_COUNT) -> None:
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "location" in record:
                record["location"] = fix_mojibake(record["location"])
            records.append(record)

    # Filter out invalid locations
    location_filtered = [r for r in records if r.get("location") in VALID_LOCATIONS]
    removed_by_location = len(records) - len(location_filtered)

    # Count removed locations for reporting
    invalid_location_counts = Counter(
        r.get("location") for r in records if r.get("location") not in VALID_LOCATIONS
    )

    title_counts = Counter(r["job_title"] for r in location_filtered)
    valid_titles = {t for t, c in title_counts.items() if c >= min_count}
    filtered = [r for r in location_filtered if r["job_title"] in valid_titles]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in filtered:
            f.write(json.dumps(rec) + "\n")

    removed_postings = len(location_filtered) - len(filtered)
    removed_titles = len(title_counts) - len(valid_titles)
    print(f"Input:   {len(records):>6} postings  {len(title_counts):>4} unique titles")
    print(f"Removed: {removed_by_location:>6} postings  (invalid location)")
    if invalid_location_counts:
        for loc, count in invalid_location_counts.most_common():
            print(f"           {count:>5}  {loc!r}")
    print(f"Removed: {removed_postings:>6} postings  {removed_titles:>4} titles (< {min_count} occurrences)")
    for title, count in sorted(title_counts.items(), key=lambda x: x[1], reverse=True):
        if title not in valid_titles:
            print(f"           {count:>5}  {title!r}")
    print(f"Output:  {len(filtered):>6} postings  {len(valid_titles):>4} unique titles")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: <input_stem>_cleaned.jsonl)")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                        help="Minimum number of occurrences to keep a job title (default: 20)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.parent / (args.input.stem + "_cleaned" + args.input.suffix)

    cleanup(args.input, args.output, args.min_count)
