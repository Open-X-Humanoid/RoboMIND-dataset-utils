#!/usr/bin/env python3
"""Scan RoboMINDv1 HDF5 datasets for puppet data quality issues.

Usage examples:
    # Focus scan: h5_ur_1rgb in benchmark1_0
    python scan_robomind_quality.py --robot-types h5_ur_1rgb --benchmarks benchmark1_0 -o ur_report.csv

    # Full scan across all benchmarks
    python scan_robomind_quality.py -o full_report.csv

    # Custom tolerance
    python scan_robomind_quality.py --ee-tolerance 0.01 -o report.csv
"""

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from robomind_data_check import (
    EpisodeQualityResult,
    QualityCheckConfig,
    check_episode,
)

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def discover_files(
    base_dirs: List[str],
    robot_types: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
) -> List[str]:
    """Walk directories to find all trajectory.hdf5 files, with filters."""
    files = []
    for base_dir in base_dirs:
        for root, dirs, fnames in os.walk(base_dir):
            if "trajectory.hdf5" in fnames:
                fpath = os.path.join(root, "trajectory.hdf5")
                if _matches_filters(fpath, robot_types, task_names, benchmarks):
                    files.append(fpath)
    return sorted(files)


def _matches_filters(fpath, robot_types, task_names, benchmarks) -> bool:
    parts = fpath.split("/")
    try:
        se_idx = parts.index("success_episodes")
    except ValueError:
        return False
    if benchmarks and se_idx >= 4 and parts[se_idx - 4] not in benchmarks:
        return False
    if robot_types and se_idx >= 3 and parts[se_idx - 3] not in robot_types:
        return False
    if task_names and se_idx >= 2 and parts[se_idx - 2] not in task_names:
        return False
    return True


# ---------------------------------------------------------------------------
# Parallel scanning
# ---------------------------------------------------------------------------
def scan_parallel(
    files: List[str],
    config: QualityCheckConfig,
    num_workers: int = 12,
    chunk_size: int = 64,
) -> List[EpisodeQualityResult]:
    worker_fn = partial(check_episode, config=config)
    results = []
    with Pool(num_workers) as pool:
        with tqdm(total=len(files), desc="Scanning", unit="file") as pbar:
            for result in pool.imap_unordered(worker_fn, files, chunksize=chunk_size):
                results.append(result)
                pbar.update(1)
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "benchmark", "robot_type", "task_name", "split", "episode_id",
    "num_frames",
    "puppet_ee_key", "puppet_ee_exists",
    "puppet_ee_max_range", "puppet_ee_is_stuck",
    "puppet_ee_range_per_dim",
    "puppet_ee_unique_count", "puppet_ee_unique_ratio",
    "puppet_jp_key", "puppet_jp_exists",
    "puppet_jp_max_range", "puppet_jp_is_stuck",
    "puppet_jp_unique_count", "puppet_jp_unique_ratio",
    "master_jp_key", "master_jp_exists",
    "master_jp_max_range", "master_jp_is_stuck",
    "master_jp_unique_count", "master_jp_unique_ratio",
    "error",
]


def _aq_to_dict(prefix: str, aq) -> dict:
    """Flatten an ArrayQualityResult (or None) into dict entries."""
    if aq is None:
        return {
            f"{prefix}_key": "",
            f"{prefix}_exists": False,
            f"{prefix}_max_range": "",
            f"{prefix}_is_stuck": "",
            f"{prefix}_unique_count": "",
            f"{prefix}_unique_ratio": "",
        }
    d = {
        f"{prefix}_key": aq.key,
        f"{prefix}_exists": aq.exists,
        f"{prefix}_max_range": f"{aq.max_range:.6f}" if aq.exists else "",
        f"{prefix}_is_stuck": aq.is_stuck,
        f"{prefix}_unique_count": aq.unique_count,
        f"{prefix}_unique_ratio": f"{aq.unique_ratio:.4f}",
    }
    if prefix == "puppet_ee":
        d[f"{prefix}_range_per_dim"] = (
            ";".join(f"{v:.6f}" for v in aq.range_per_dim)
            if aq.range_per_dim else ""
        )
    return d


def result_to_row(r: EpisodeQualityResult) -> dict:
    row = {
        "benchmark": r.benchmark,
        "robot_type": r.robot_type,
        "task_name": r.task_name,
        "split": r.split,
        "episode_id": r.episode_id,
        "num_frames": r.num_frames,
        "error": r.error or "",
    }
    row.update(_aq_to_dict("puppet_ee", r.puppet_ee))
    row.update(_aq_to_dict("puppet_jp", r.puppet_jp))
    row.update(_aq_to_dict("master_jp", r.master_jp))
    return row


def write_csv(results: List[EpisodeQualityResult], output_path: str) -> pd.DataFrame:
    rows = [result_to_row(r) for r in results]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"\nResults written to {output_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def print_summary(results: List[EpisodeQualityResult]):
    rows = [result_to_row(r) for r in results]
    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("PUPPET END EFFECTOR QUALITY REPORT")
    print("=" * 70)

    # --- Puppet EE ---
    df_ee = df[df["puppet_ee_exists"] == True].copy()
    if len(df_ee) == 0:
        print("\nNo files with puppet end effector data found.")
    else:
        df_ee["puppet_ee_is_stuck"] = df_ee["puppet_ee_is_stuck"].astype(bool)
        total = len(df_ee)
        stuck = df_ee["puppet_ee_is_stuck"].sum()
        print(f"\nPuppet EE stuck: {stuck}/{total} ({100*stuck/total:.1f}%)")

        print("\n--- By Robot Type ---")
        for robot, grp in df_ee.groupby("robot_type"):
            n = len(grp)
            b = grp["puppet_ee_is_stuck"].sum()
            print(f"  {robot}: {b}/{n} ({100*b/n:.1f}%)")

        print("\n--- By Benchmark ---")
        for bm, grp in df_ee.groupby("benchmark"):
            n = len(grp)
            b = grp["puppet_ee_is_stuck"].sum()
            print(f"  {bm}: {b}/{n} ({100*b/n:.1f}%)")

        print("\n--- Worst Tasks (top 30 by % stuck EE) ---")
        task_stats = (
            df_ee.groupby(["benchmark", "robot_type", "task_name"])
            .agg(total=("puppet_ee_is_stuck", "count"),
                 stuck=("puppet_ee_is_stuck", "sum"))
            .reset_index()
        )
        task_stats["pct"] = 100 * task_stats["stuck"] / task_stats["total"]
        task_stats = task_stats[task_stats["stuck"] > 0].sort_values(
            "pct", ascending=False
        )
        for _, row in task_stats.head(30).iterrows():
            print(
                f"  {row['benchmark']}/{row['robot_type']}/{row['task_name']}: "
                f"{row['stuck']}/{row['total']} ({row['pct']:.0f}%)"
            )

    # --- Puppet JP ---
    df_jp = df[df["puppet_jp_exists"] == True].copy()
    if len(df_jp) > 0:
        df_jp["puppet_jp_is_stuck"] = df_jp["puppet_jp_is_stuck"].astype(bool)
        total_jp = len(df_jp)
        stuck_jp = df_jp["puppet_jp_is_stuck"].sum()
        print(f"\n--- Puppet JP stuck: {stuck_jp}/{total_jp} ({100*stuck_jp/total_jp:.1f}%) ---")

    # --- Errors ---
    errors = df[df["error"] != ""]
    if len(errors) > 0:
        print(f"\n--- Errors: {len(errors)} files ---")
        for err_type, cnt in errors["error"].value_counts().head(10).items():
            print(f"  [{cnt}] {err_type}")

    # --- Files without puppet EE ---
    no_ee = df[df["puppet_ee_exists"] == False]
    if len(no_ee) > 0:
        print(f"\n--- Files without puppet/end_effector: {len(no_ee)} ---")
        for robot, grp in no_ee.groupby("robot_type"):
            print(f"  {robot}: {len(grp)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scan RoboMINDv1 HDF5 datasets for puppet data quality issues."
    )
    parser.add_argument(
        "--base-dirs", nargs="+",
        default=["/jushen/xr-2/gongda/datasets/hf_robomindv1"],
        help="Root directories to scan",
    )
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--robot-types", nargs="*", default=None)
    parser.add_argument("--task-names", nargs="*", default=None)
    parser.add_argument("-o", "--output", default="robomind_quality_report.csv")
    parser.add_argument("-w", "--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--ee-tolerance", type=float, default=1e-3,
        help="Max range below which puppet EE is considered stuck (default: 1e-3)",
    )
    parser.add_argument(
        "--jp-tolerance", type=float, default=1e-3,
        help="Max range below which puppet JP is considered stuck (default: 1e-3)",
    )
    parser.add_argument("--no-range", action="store_true",
                        help="Skip per-dim range in output")

    args = parser.parse_args()

    config = QualityCheckConfig(
        ee_tolerance=args.ee_tolerance,
        jp_tolerance=args.jp_tolerance,
        compute_range_per_dim=not args.no_range,
    )

    print("Discovering trajectory.hdf5 files...")
    files = discover_files(
        args.base_dirs,
        robot_types=args.robot_types,
        task_names=args.task_names,
        benchmarks=args.benchmarks,
    )
    print(f"Found {len(files)} files")

    if not files:
        print("No files found. Check your --base-dirs and filters.")
        return

    results = scan_parallel(files, config, args.workers, args.chunk_size)
    write_csv(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
