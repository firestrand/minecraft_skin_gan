"""Enforce the user contract independently for line and branch coverage."""

import json
from pathlib import Path


def main() -> None:
    report = json.loads(Path("artifacts/coverage.json").read_text())
    totals = report["totals"]
    for label, covered, total in (
        ("line", totals["covered_lines"], totals["num_statements"]),
        ("branch", totals["covered_branches"], totals["num_branches"]),
    ):
        percentage = 100 * covered / total if total else 100.0
        print(f"{label} coverage: {percentage:.2f}% ({covered}/{total})")
        if percentage <= 80:
            raise SystemExit(f"{label} coverage must exceed 80%")


if __name__ == "__main__":
    main()
