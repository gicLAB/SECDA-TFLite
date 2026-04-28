#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- robust bash array parsing ----------

ARRAYS_NEEDED = [
    "hw_array", "tag_array", "app_array", "model_array",
    "cmd_array", "del_array", "del_version_array", "version_array",
]
DECL_RE = re.compile(r'declare\s+-a\s+(\w+)\s*=\s*\(\s*(.*?)\s*\)', re.DOTALL)


def bash_array_tokenize(body: str) -> List[str]:
    """
    Tokenize bash array body like:  ("a" "b"  "x"y"z"  foo  "bar baz")
    - Handles double/single quotes
    - Concatenates adjacent quoted/unquoted segments into one element
    - Splits elements only on whitespace *outside* quotes
    """
    items: List[str] = []
    curr: List[str] = []
    i, n = 0, len(body)
    state = "out"   # out | dq | sq
    while i < n:
        ch = body[i]
        if state == "out":
            if ch.isspace():
                if curr:
                    items.append("".join(curr))
                    curr = []
                i += 1
                continue
            if ch == '"':
                state = "dq"
                i += 1
                continue
            if ch == "'":
                state = "sq"
                i += 1
                continue
            curr.append(ch)
            i += 1
            continue
        elif state == "dq":
            if ch == '\\' and i+1 < n:
                curr.append(body[i+1])
                i += 2
                continue
            if ch == '"':
                state = "out"
                i += 1
                continue
            curr.append(ch)
            i += 1
            continue
        else:  # sq
            if ch == "'":
                state = "out"
                i += 1
                continue
            curr.append(ch)
            i += 1
            continue
    if curr:
        items.append("".join(curr))
    # collapse empty items that come from formatting
    return [s for s in items if s != ""]


def parse_bash_arrays(text: str) -> Dict[str, List[str]]:
    arrays: Dict[str, List[str]] = {}
    for name, body in DECL_RE.findall(text):
        arrays[name] = bash_array_tokenize(body)
    return arrays


def load_arrays_from_config(cfg: Path) -> Dict[str, List[str]]:
    txt = cfg.read_text(encoding="utf-8", errors="ignore")
    arrays = parse_bash_arrays(txt)
    missing = [a for a in ARRAYS_NEEDED if a not in arrays]
    if missing:
        print(
            f"⚠️  Missing arrays in {cfg.name}: {', '.join(missing)}", file=sys.stderr)
    return arrays


# ---------- threads/runs from cmd ----------
THREAD_RE = re.compile(r'--num_threads\s*=\s*(\d+)')
RUNS_RE = re.compile(r'--num_runs\s*=\s*(\d+)')


def parse_threads_runs(cmd: str):
    th = THREAD_RE.search(cmd or "")
    rn = RUNS_RE.search(cmd or "")
    return (int(th.group(1)) if th else None, int(rn.group(1)) if rn else None)

# ---------- main ----------


def main():
    ap = argparse.ArgumentParser(
        description="Index SECDA results by tag_array and emit index.csv")
    ap.add_argument("--results-dir", required=True,
                    help="Path to ./results/<name>")
    ap.add_argument("--config", required=True,
                    help="Path to generated config.sh")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    cfg_path = Path(args.config).resolve()
    if not results_dir.is_dir():
        sys.exit(f"❌ Results dir not found: {results_dir}")
    if not cfg_path.is_file():
        sys.exit(f"❌ config.sh not found: {cfg_path}")

    arr = load_arrays_from_config(cfg_path)
    tags = arr.get("tag_array", [])
    hws = arr.get("hw_array", [])
    apps = arr.get("app_array", [])
    models = arr.get("model_array", [])
    cmds = arr.get("cmd_array", [])
    dels = arr.get("del_array", [])
    delvs = arr.get("del_version_array", [])
    vers = arr.get("version_array", [])

    rows = []
    for i, tag in enumerate(tags):
        experiment = tag
        power_name = f"{experiment}_power.txt"
        latency_name = f"{experiment}.txt"

        app_name = apps[i] if i < len(apps) else ""
        hw_name = hws[i] if i < len(hws) else ""
        model_name = models[i] if i < len(models) else ""
        delegate_name = dels[i] if i < len(dels) else ""
        delegate_version = delvs[i] if i < len(delvs) else ""
        version = vers[i] if i < len(vers) else ""
        cmd = cmds[i] if i < len(cmds) else ""
        thread_count, number_run = parse_threads_runs(cmd)

        rows.append({
            "experiment": experiment,
            "power_file_name": power_name,
            "latency_file_name": latency_name,
            "app_name": app_name,
            "hw_name": hw_name,
            "delegate_name": delegate_name,
            "delegate_version": delegate_version,
            "model_name": model_name,
            "thread_count": thread_count if thread_count is not None else "",
            "number_run": number_run if number_run is not None else "",
            "version": version,
        })

    out_csv = results_dir / "index.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "experiment", "power_file_name", "latency_file_name",
            "app_name", "hw_name", "delegate_name", "delegate_version",
            "model_name", "thread_count", "number_run", "version"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"✅ Wrote {out_csv}")


if __name__ == "__main__":
    main()
