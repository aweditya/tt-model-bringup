#!/usr/bin/env python3
"""Summarize qb2 Tracy trace-replay overlap artifacts.

This parser is intentionally conservative.  It reports only facts that are
available in the raw Tracy CSVs: synchronized per-device trace spans and the
extent of host op metadata coverage.  It does not infer per-op replay timing
when the Tenstorrent postprocessor fails to join host ops to device rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


def _read_sync(log_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    scales: dict[str, float] = {}
    shifts: dict[str, float] = {}
    sync_path = log_dir / "sync_device_info.csv"
    if not sync_path.exists():
        return scales, shifts

    with sync_path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            dev = row.get("device id", "").strip()
            scale = row.get("device_frequency_ratio", "").strip()
            shift = row.get("device_shift", "").strip()
            if dev and scale and shift:
                scales[dev] = float(scale)
                shifts[dev] = float(shift)
    return scales, shifts


def _paired_trace_fw_intervals(log_dir: Path, scales: dict[str, float], shifts: dict[str, float]) -> list[dict]:
    profile_path = log_dir / "profile_log_device.csv"
    intervals: list[dict] = []
    starts: dict[tuple[str, str, str, str, str], list[float]] = {}

    with profile_path.open(newline="") as f:
        f.readline()  # ARCH line
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            slot = row["PCIe slot"].strip()
            zone = row["zone name"].strip()
            if zone != "TRACE-FW":
                continue
            key = (
                slot,
                row["core_x"].strip(),
                row["core_y"].strip(),
                row["run host ID"].strip(),
                zone,
            )
            timestamp = float(row["time[cycles since reset]"])
            timestamp = timestamp * scales.get(slot, 1.0) + shifts.get(slot, 0.0)
            if row["type"].strip() == "ZONE_START":
                starts.setdefault(key, []).append(timestamp)
            elif row["type"].strip() == "ZONE_END" and starts.get(key):
                start = starts[key].pop()
                intervals.append({
                    "slot": slot,
                    "run": key[3],
                    "start_cycles": start,
                    "end_cycles": timestamp,
                    "duration_ms": (timestamp - start) / 1350.0 / 1000.0,
                })
    return intervals


def _summarize_trace_fw(intervals: list[dict]) -> list[dict]:
    rows: list[dict] = []
    runs = sorted({i["run"] for i in intervals if i["run"]}, key=lambda value: int(value))
    slots = sorted({i["slot"] for i in intervals}, key=lambda value: int(value))
    for run in runs:
        per_slot = []
        for slot in slots:
            vals = [i for i in intervals if i["run"] == run and i["slot"] == slot]
            if not vals:
                continue
            start = statistics.median(i["start_cycles"] for i in vals)
            end = statistics.median(i["end_cycles"] for i in vals)
            duration = statistics.median(i["duration_ms"] for i in vals)
            per_slot.append({
                "slot": slot,
                "core_count": len(vals),
                "median_start_cycles": start,
                "median_end_cycles": end,
                "median_duration_ms": duration,
            })
        if not per_slot:
            continue
        min_start = min(item["median_start_cycles"] for item in per_slot)
        max_start = max(item["median_start_cycles"] for item in per_slot)
        min_end = min(item["median_end_cycles"] for item in per_slot)
        max_end = max(item["median_end_cycles"] for item in per_slot)
        rows.append({
            "run": run,
            "slots": [
                {
                    "slot": item["slot"],
                    "core_count": item["core_count"],
                    "start_offset_ms": (item["median_start_cycles"] - min_start) / 1350.0 / 1000.0,
                    "end_offset_ms": (item["median_end_cycles"] - min_start) / 1350.0 / 1000.0,
                    "median_duration_ms": item["median_duration_ms"],
                }
                for item in per_slot
            ],
            "max_start_skew_ms": (max_start - min_start) / 1350.0 / 1000.0,
            "max_end_skew_ms": (max_end - min_end) / 1350.0 / 1000.0,
        })
    return rows


def _summarize_host_op_metadata(log_dir: Path) -> dict:
    op_data_path = log_dir / "tracy_ops_data.csv"
    op_times_path = log_dir / "tracy_ops_times.csv"
    id_to_name: dict[str, str] = {}
    pattern = re.compile(r'TT_DNN_DEVICE_OP: "([^"]+)", [^,]+, [^,]+, [^,]+, ([0-9]+) ->')

    if op_data_path.exists():
        with op_data_path.open(errors="replace") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    id_to_name[match.group(2)] = match.group(1)

    rows = 0
    unknown = 0
    counts: Counter[str] = Counter()
    if op_times_path.exists():
        with op_times_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["name"] != "TT_DNN_DEVICE_OP":
                    continue
                rows += 1
                op_id = row["zone_text"].split(":", 1)[-1]
                name = id_to_name.get(op_id)
                if name is None:
                    unknown += 1
                else:
                    counts[name] += 1

    return {
        "known_op_metadata_ids": len(id_to_name),
        "tt_dnn_device_op_time_rows": rows,
        "unknown_time_rows": unknown,
        "known_time_rows_by_op": counts.most_common(40),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path, help="Directory containing Tracy .logs CSV files")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    scales, shifts = _read_sync(args.log_dir)
    intervals = _paired_trace_fw_intervals(args.log_dir, scales, shifts)
    runs = _summarize_trace_fw(intervals)
    result = {
        "log_dir": str(args.log_dir),
        "sync_available": bool(scales and shifts),
        "sync_scales": scales,
        "sync_shifts": shifts,
        "trace_fw_runs": runs,
        "host_op_metadata": _summarize_host_op_metadata(args.log_dir),
        "limitations": [
            "TRACE-FW spans show coarse per-chip replay overlap after applying sync scale/shift.",
            "These CSVs do not by themselves label matmul versus collective intervals inside execute_trace.",
            "Host TT_DNN_DEVICE_OP rows are capture/build-time host zones; replayed trace internals need a successful enriched report or extra device/NOC annotations.",
        ],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"sync_available: {result['sync_available']}")
        for row in runs:
            print(
                f"run {row['run']}: max_start_skew={row['max_start_skew_ms']:.6f} ms "
                f"max_end_skew={row['max_end_skew_ms']:.6f} ms"
            )
            for slot in row["slots"]:
                print(
                    f"  slot {slot['slot']}: start+{slot['start_offset_ms']:.6f} ms "
                    f"duration={slot['median_duration_ms']:.6f} ms"
                )
        meta = result["host_op_metadata"]
        print(
            "host_op_metadata: "
            f"{meta['known_op_metadata_ids']} known ids, "
            f"{meta['tt_dnn_device_op_time_rows']} timing rows, "
            f"{meta['unknown_time_rows']} unknown timing rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
