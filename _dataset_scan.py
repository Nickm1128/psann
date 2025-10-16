import csv
from pathlib import Path
from datetime import datetime
import json
import sys

ROOT = Path("datasets")

datetime_patterns = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d"
]

missing_tokens = {"", "NA", "NAN", "NULL", "None", "na", "nan"}

def try_parse_datetime(value):
    raw = value.strip()
    if not raw:
        return None
    # normalize decimal comma in timestamps
    if "," in raw and raw.count(",") == 1 and raw.split(",")[0].isdigit():
        # likely decimal comma in seconds
        raw = raw.replace(",", ".")
    elif "," in raw and raw.count(",") > 1:
        # replace comma between date/time components e.g. 2016-... 18:31:46,003
        parts = raw.rsplit(",", 1)
        if len(parts) == 2 and parts[0].replace("-", "").replace(":", "").replace(" ", "").isdigit():
            raw = parts[0] + "." + parts[1]
    for fmt in datetime_patterns:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None

def analyze_csv(path: Path):
    try:
        fh = path.open("r", encoding="utf-8", newline="")
    except UnicodeDecodeError:
        fh = path.open("r", encoding="latin-1", newline="")
    with fh as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "path": path.as_posix(),
                "rows": 0,
                "columns": [],
                "datetime_columns": []
            }
        total = 0
        dt_indices = [idx for idx, name in enumerate(header) if any(key in name.lower() for key in ("date", "time"))]
        dt_min = [None] * len(dt_indices)
        dt_max = [None] * len(dt_indices)
        for row in reader:
            total += 1
            for pos, col_idx in enumerate(dt_indices):
                if col_idx >= len(row):
                    continue
                val = row[col_idx].strip()
                if not val or val in missing_tokens:
                    continue
                dt = try_parse_datetime(val)
                if not dt:
                    continue
                if dt_min[pos] is None or dt < dt_min[pos]:
                    dt_min[pos] = dt
                if dt_max[pos] is None or dt > dt_max[pos]:
                    dt_max[pos] = dt
        return {
            "path": path.as_posix(),
            "rows": total,
            "columns": header,
            "datetime_columns": [
                {
                    "name": header[col_idx],
                    "min": dt_min[pos].isoformat() if dt_min[pos] else None,
                    "max": dt_max[pos].isoformat() if dt_max[pos] else None
                }
                for pos, col_idx in enumerate(dt_indices)
            ]
        }

def main():
    reports = []
    for csv_path in sorted(ROOT.rglob("*.csv")):
        reports.append(analyze_csv(csv_path))
    json.dump(reports, sys.stdout, indent=2)

if __name__ == "__main__":
    main()
