from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_profile_subprocess(profile_name: str) -> str:
    cmd = [sys.executable, "main.py", "--profile", profile_name]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print("\n" + "=" * 90)
    print(f"RUNNING PROFILE: {profile_name}")
    print("=" * 90)

    if completed.stdout:
        print(completed.stdout)

    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr)
        raise RuntimeError(f"Profile '{profile_name}' failed with exit code {completed.returncode}.")

    return completed.stdout


def _extract_float(text: str, label: str):
    pattern = rf"{re.escape(label)}:\s+(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _extract_int(text: str, label: str):
    pattern = rf"{re.escape(label)}:\s+(-?\d+)"
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _build_summary_row(profile_name: str, stdout: str) -> Dict:
    return {
        "profile": profile_name,
        "final_equity": _extract_float(stdout, "Final equity"),
        "total_return_pct": _extract_float(stdout, "Total return (%)"),
        "trade_profit_factor": _extract_float(stdout, "Trade profit factor"),
        "max_drawdown_pct": _extract_float(stdout, "Max drawdown (%)"),
        "trade_lifecycles": _extract_int(stdout, "Trade lifecycles"),
        "avg_open_positions": _extract_float(stdout, "avg_open_positions"),
        "pct_time_with_positions": _extract_float(stdout, "pct_time_with_positions"),
    }


def run_profile(profile_name: str) -> Dict:
    stdout = _run_profile_subprocess(profile_name)
    return _build_summary_row(profile_name, stdout)


def run_all_profiles(profile_names: Optional[List[str]] = None) -> pd.DataFrame:
    if profile_names is None:
        profile_names = [
            "trend_only",
            "trend_plus_breakout",
            "full_system",
            "next_research",
        ]

    rows = [run_profile(profile_name) for profile_name in profile_names]

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        numeric_cols = [
            "final_equity",
            "total_return_pct",
            "trade_profit_factor",
            "max_drawdown_pct",
            "avg_open_positions",
            "pct_time_with_positions",
        ]
        for col in numeric_cols:
            if col in comparison.columns:
                comparison[col] = pd.to_numeric(comparison[col], errors="coerce").round(4)

    print("\n" + "=" * 90)
    print("PROFILE COMPARISON")
    print("=" * 90)
    if comparison.empty:
        print("No results.")
    else:
        print(comparison.to_string(index=False))

    return comparison


if __name__ == "__main__":
    run_all_profiles()
