"""
Counts occurrences of each (job_title, location) pair in a JSONL file
and writes a ranked report.

Usage:
    python data_preprocessing/title_location_pairs.py
    python data_preprocessing/title_location_pairs.py --input data/mycustom.jsonl

Output:
    data_preprocessing/report_title_location_pairs.txt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("data/structured_jobs_20.05_normalized_cleaned.jsonl")
OUTPUT_PATH = Path("data_preprocessing/report_title_location_pairs.txt")


def title_location_pairs(input_path: Path, output_path: Path) -> None:
    counts: Counter = Counter()
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts[(rec["job_title"], rec["location"])] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"(job_title, location) pair counts\n")
        f.write(f"Source: {input_path}\n")
        f.write(f"Total records: {sum(counts.values())}\n")
        f.write(f"Unique pairs:  {len(counts)}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Count':>6}  {'Job Title':<40}  Location\n")
        f.write("-" * 75 + "\n")
        for (title, location), count in counts.most_common():
            f.write(f"{count:>6}  {title:<40}  {location}\n")

    print(f"{len(counts)} unique (title, location) pairs → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    title_location_pairs(args.input, args.output)
