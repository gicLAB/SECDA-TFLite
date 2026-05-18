#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Optional

# Regexes for average time
AVG_MS_RE = re.compile(
    r'average time:\s*([0-9]*\.?[0-9]+)\s*ms', re.IGNORECASE)
AVG_US_KV_RE = re.compile(
    r'\bavg\s*=\s*([0-9]+)\b', re.IGNORECASE)  # count=.. avg=163758
INFER_AVG_US_RE = re.compile(
    r'Inference\s*\(avg\)\s*:\s*([0-9]+)', re.IGNORECASE)

# Keys we care about
COUNTER_KEYS = [
    "load_send_inputs", "load_send_weights", "send_weights", "set_results",
    "start_compute", "receive_results", "store", "vm_acc", "ipack", "conv_total",
]

COUNTER_LINE_RE = re.compile(
    r'^\s*([a-zA-Z0-9_]+)\s*:\s*([0-9]+)\s*$', re.MULTILINE)


def parse_total_latency(txt: str) -> Optional[float]:
    """Extract total latency in ms if present."""
    m = AVG_MS_RE.search(txt)
    if m:
        return float(m.group(1))
    m = INFER_AVG_US_RE.search(txt)
    if m:
        return float(m.group(1)) / 1000.0
    m = AVG_US_KV_RE.search(txt)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def parse_counters_ms(txt: str, number_run: int) -> Dict[str, str]:
    """Return counters divided by (1000*number_run) as ms, or empty string if not found."""
    denom = 1000.0 * (number_run if number_run else 1)
    found = {k: "" for k in COUNTER_KEYS}
    for key, val in COUNTER_LINE_RE.findall(txt):
        if key in found:
            ms = int(val) / denom
            found[key] = f"{ms:.3f}"
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Augment index.csv with latency results")
    ap.add_argument("--results-dir", required=True,
                    help="Path to ./results/<name>")
    ap.add_argument("--index",       required=True,
                    help="Path to index.csv produced earlier")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    index_csv = Path(args.index).resolve()
    if not results_dir.is_dir():
        sys.exit(f"❌ Results dir not found: {results_dir}")
    if not index_csv.is_file():
        sys.exit(f"❌ index.csv not found: {index_csv}")

    # Read original index.csv
    with index_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Extend fieldnames
    if "Total_latency" not in fieldnames:
        fieldnames.append("Total_latency")
    for k in COUNTER_KEYS:
        if k not in fieldnames:
            fieldnames.append(k)
    if "CPU_total" not in fieldnames:
        fieldnames.append("CPU_total")

    # Update rows
    for r in rows:
        latfn = r.get("latency_file_name", "")
        try:
            runs = int(r.get("number_run", ""))
        except:
            runs = 1

        lat_path = results_dir / latfn
        if not latfn or not lat_path.is_file():
            r["Total_latency"] = ""
            for k in COUNTER_KEYS:
                r[k] = ""
            r["CPU_total"] = ""
            continue

        txt = lat_path.read_text(errors="ignore")
        total_latency = parse_total_latency(txt)
        counters = parse_counters_ms(txt, runs)

        r["Total_latency"] = f"{total_latency:.3f}" if total_latency is not None else ""
        for k in COUNTER_KEYS:
            r[k] = counters[k]

        # CPU_total = Total_latency - conv_total
        try:
            conv = float(r.get("conv_total", "")) if r.get(
                "conv_total") else 0.0
            tl = float(r["Total_latency"]) if r["Total_latency"] else 0.0
            if tl and conv:
                r["CPU_total"] = f"{tl - conv:.3f}"
            else:
                r["CPU_total"] = ""
        except Exception:
            r["CPU_total"] = ""

    # Reorder columns: Total_latency, conv_total, CPU_total immediately after
    def reorder(fields):
        out = []
        for f in fields:
            if f not in ("Total_latency", "conv_total", "CPU_total"):
                out.append(f)
        # append our desired order at the end
        out += ["Total_latency", "conv_total", "CPU_total"]
        return out

    new_fieldnames = reorder(fieldnames)

    # Rewrite index.csv
    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"✅ index.csv updated with Total_latency, conv_total, and CPU_total at {index_csv}")


if __name__ == "__main__":
    main()
