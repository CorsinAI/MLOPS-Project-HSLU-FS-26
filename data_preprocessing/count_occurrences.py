"""
Counts occurrences of job titles and/or locations in a JSONL file.
Writes a separate report file for each requested field.

Usage:
    python data_preprocessing/count_occurrences.py --job_title
    python data_preprocessing/count_occurrences.py --location
    python data_preprocessing/count_occurrences.py --job_title --location
    python data_preprocessing/count_occurrences.py --job_title --input data/mycustom.jsonl

Output files (written to data_preprocessing/):
    report_job_titles.txt
    report_locations.txt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("data/structured_jobs_20.05_normalized_cleaned.jsonl")
OUTPUT_DIR = Path("data_preprocessing")

FIELD_TO_REPORT = {
    "job_title": OUTPUT_DIR / "report_job_titles.txt",
    "location":  OUTPUT_DIR / "report_locations.txt",
}


def count_and_report(input_path: Path, field: str, output_path: Path) -> None:
    counts: Counter = Counter()
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts[rec[field]] += 1

    with open(output_path, "w") as f:
        f.write(f"Occurrences by {field}\n")
        f.write(f"Source: {input_path}\n")
        f.write(f"Total records: {sum(counts.values())}\n")
        f.write(f"Unique values: {len(counts)}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Count':>6}  Value\n")
        f.write("-" * 55 + "\n")
        for value, count in counts.most_common():
            f.write(f"{count:>6}  {value}\n")

    print(f"{field}: {len(counts)} unique values → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--job_title", action="store_true", help="Count occurrences by job title")
    parser.add_argument("--location", action="store_true", help="Count occurrences by location")
    args = parser.parse_args()

    if not args.job_title and not args.location:
        parser.error("Specify at least one of --job_title or --location")

    if args.job_title:
        count_and_report(args.input, "job_title", FIELD_TO_REPORT["job_title"])

    if args.location:
        count_and_report(args.input, "location", FIELD_TO_REPORT["location"])
