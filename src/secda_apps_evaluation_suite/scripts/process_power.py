#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path


def read_avg_power_w(path: Path) -> float | None:
    """Read a power text file with optional 'power_uW' header and return average power in Watts."""
    if not path.is_file():
        return None
    lines = path.read_text(errors="ignore").strip().splitlines()
    if not lines:
        return None
    # Skip header if present
    if lines and lines[0].strip().lower().endswith("uw"):
        lines = lines[1:]
    vals = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            vals.append(float(ln))  # in microWatts
        except ValueError:
            # ignore non-numeric lines
            pass
    if not vals:
        return None
    avg_uW = sum(vals) / len(vals)
    return avg_uW / 1_000_000.0  # -> Watts


def main():
    ap = argparse.ArgumentParser(
        description="Augment index.csv with Power_inference, Power_idle, and Total_power (Watts)"
    )
    ap.add_argument("--results-dir", required=True,
                    help="Path to ./results/<name>")
    ap.add_argument("--index",       required=True, help="Path to index.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    index_csv = Path(args.index).resolve()
    if not results_dir.is_dir():
        sys.exit(f"❌ Results dir not found: {results_dir}")
    if not index_csv.is_file():
        sys.exit(f"❌ index.csv not found: {index_csv}")

    # Load idle power once
    idle_file = results_dir / "cpu_idle_power.txt"
    power_idle = read_avg_power_w(idle_file)

    # Load index
    with index_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Ensure columns exist (Power_inference replaces prior Total_power meaning)
    for col in ("Power_inference", "Power_idle", "Total_power"):
        if col not in fieldnames:
            fieldnames.append(col)

    # Update rows
    for r in rows:
        # Per-experiment inference power
        power_name = r.get("power_file_name", "") or ""
        p_path = results_dir / power_name if power_name else None
        p_inf = read_avg_power_w(p_path) if power_name else None

        # Write Power_inference
        r["Power_inference"] = f"{p_inf:.6f}" if p_inf is not None else ""

        # Write Power_idle (same for all rows; if missing, leave empty)
        r["Power_idle"] = f"{power_idle:.6f}" if power_idle is not None else ""

        # Total_power = Power_inference - Power_idle
        try:
            if p_inf is not None and power_idle is not None:
                total = p_inf - power_idle
                r["Total_power"] = f"{total:.6f}"
            else:
                r["Total_power"] = ""
        except Exception:
            r["Total_power"] = ""

    # Write back
    with index_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    msg_idle = f"{power_idle:.6f} W" if power_idle is not None else "N/A"
    print(
        f"✅ index.csv updated with Power_inference, Power_idle ({msg_idle}), and Total_power at {index_csv}")


if __name__ == "__main__":
    main()
