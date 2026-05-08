"""Sara Evaluation Benchmark.

Loads a dataset of natural-language commands with expected structured actions,
runs each command through the full LLM parsing pipeline, optionally executes
them against real Excel files, and produces a summary report covering:

    - Intent parsing accuracy
    - Excel execution success rate
    - LLM latency (avg, P50, P95)
    - LLM fallback frequency (Groq -> Ollama -> heuristic)

Usage::

    python evaluate.py                                    # default dataset
    python evaluate.py --dataset tests/command_dataset.json
    python evaluate.py --execute                          # also run Excel actions
    python evaluate.py --report report.json --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import tempfile
import time

# Ensure backend imports work when script is run directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    load_dotenv(_env, override=True)

from llm_router import route_llm
from excel_controller import execute_action, create_excel
from metrics import metrics

log = logging.getLogger("evaluate")

# Path to the shipped dataset
_DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tests", "command_dataset.json"
)


# --------------- helpers ---------------


def _action_type(result: dict) -> str:
    """Normalise the effective type from an LLM-parsed result."""
    if isinstance(result.get("steps"), list) and result["steps"]:
        return "steps"
    return result.get("type", "unknown") or "unknown"


def _partial_match(expected: dict, actual: dict) -> dict:
    """Compare each key in *expected* against *actual*.

    Returns ``{"matched": [...], "mismatched": [...]}`` so the report can
    show *which* fields diverged.
    """
    matched, mismatched = [], []
    for key, want in expected.items():
        got = actual.get(key)

        # Type field needs special handling for "steps"
        if key == "type":
            effective = _action_type(actual)
            if str(want).lower() == effective.lower():
                matched.append(key)
            else:
                mismatched.append({"field": key, "expected": want, "actual": effective})
            continue

        if _values_equal(want, got):
            matched.append(key)
        else:
            mismatched.append({"field": key, "expected": want, "actual": got})
    return {"matched": matched, "mismatched": mismatched}


def _values_equal(want, got) -> bool:
    """Loosely compare expected vs actual values."""
    if want is None and got is None:
        return True
    if isinstance(want, bool) or isinstance(got, bool):
        return want == got
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return False
        return all(_values_equal(w, g) for w, g in zip(want, got))
    # Numeric comparison
    try:
        if float(want) == float(got):
            return True
    except (TypeError, ValueError):
        pass
    # Case-insensitive string comparison
    return str(want).strip().lower() == str(got).strip().lower()


# --------------- core evaluation ---------------


def evaluate_single(
    case: dict,
    *,
    execute: bool = False,
    work_dir: str | None = None,
) -> dict:
    """Run one benchmark case.

    Parameters
    ----------
    case : dict
        Must have ``command`` (str) and ``expected`` (dict).
    execute : bool
        If True, also run the parsed action through `execute_action`.
    work_dir : str | None
        Temp directory for Excel execution (only when *execute* is True).
    """
    text = case["command"]
    expected = case["expected"]
    expected_type = expected.get("type", "unknown")

    # --- LLM parse ---
    t0 = time.perf_counter()
    result = route_llm(text)
    parse_latency_ms = (time.perf_counter() - t0) * 1000

    actual_type = _action_type(result)
    type_match = actual_type.lower() == expected_type.lower()
    field_report = _partial_match(expected, result)
    fields_pass = len(field_report["mismatched"]) == 0

    # --- optional Excel execution ---
    exec_result = None
    exec_success = None
    exec_latency_ms = 0.0
    if execute and type_match and actual_type not in ("unknown", "steps"):
        filepath = os.path.join(work_dir or tempfile.gettempdir(), "eval_test.xlsx")
        # Create a seed workbook so actions have something to operate on
        if not os.path.exists(filepath):
            create_excel(filepath, headers=["Name", "Score", "Grade"])
            # Add a couple of seed rows
            from excel_controller import add_row
            add_row(filepath, ["Seed", 1, "A"])
            add_row(filepath, ["Data", 2, "B"])
        t1 = time.perf_counter()
        try:
            exec_result = execute_action(result, filepath)
            exec_latency_ms = (time.perf_counter() - t1) * 1000
            exec_success = not (isinstance(exec_result, dict) and "error" in exec_result)
        except Exception as exc:
            exec_latency_ms = (time.perf_counter() - t1) * 1000
            exec_result = {"error": str(exc)}
            exec_success = False

    return {
        "command": text,
        "expected_type": expected_type,
        "actual_type": actual_type,
        "type_match": type_match,
        "fields_pass": fields_pass,
        "field_detail": field_report,
        "overall_pass": type_match and fields_pass,
        "parse_latency_ms": round(parse_latency_ms, 2),
        "exec_success": exec_success,
        "exec_latency_ms": round(exec_latency_ms, 2),
        "parsed_action": result,
        "exec_result": exec_result,
    }


def run_benchmark(
    dataset: list[dict],
    *,
    execute: bool = False,
    verbose: bool = True,
) -> dict:
    """Execute the full benchmark suite and return a summary report."""
    total = len(dataset)

    # Reset metrics so the summary reflects only this benchmark run
    metrics.reset()

    # Temp dir for execution tests
    work_dir = None
    if execute:
        work_dir = tempfile.mkdtemp(prefix="sara_eval_")

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"  Sara Evaluation Benchmark — {total} test cases")
        if execute:
            print(f"  Excel execution ENABLED  (work dir: {work_dir})")
        print(f"{'=' * 72}\n")

    results: list[dict] = []
    for i, case in enumerate(dataset, 1):
        r = evaluate_single(case, execute=execute, work_dir=work_dir)
        results.append(r)
        if verbose:
            tag = "\u2713 PASS" if r["overall_pass"] else "\u2717 FAIL"
            exec_tag = ""
            if execute and r["exec_success"] is not None:
                exec_tag = "  exec=" + ("\u2713" if r["exec_success"] else "\u2717")
            print(
                f"  [{i:2d}] {tag}  {r['parse_latency_ms']:7.1f}ms  "
                f"expected={r['expected_type']:20s}  got={r['actual_type']:20s}"
                f"{exec_tag}  | {r['command'][:50]}"
            )

    # Clean up temp dir
    if work_dir and os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)

    # ---- aggregate metrics ----
    passed = sum(1 for r in results if r["overall_pass"])
    type_correct = sum(1 for r in results if r["type_match"])
    latencies = sorted(r["parse_latency_ms"] for r in results)
    avg_lat = sum(latencies) / total if total else 0
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

    exec_tested = [r for r in results if r["exec_success"] is not None]
    exec_passed = sum(1 for r in exec_tested if r["exec_success"])

    # Per-type breakdown
    type_stats: dict[str, dict] = {}
    for r in results:
        t = r["expected_type"]
        bucket = type_stats.setdefault(t, {"total": 0, "correct": 0, "latencies": []})
        bucket["total"] += 1
        if r["type_match"]:
            bucket["correct"] += 1
        bucket["latencies"].append(r["parse_latency_ms"])
    per_type = {}
    for t, b in type_stats.items():
        per_type[t] = {
            "total": b["total"],
            "correct": b["correct"],
            "accuracy": round(b["correct"] / b["total"], 4) if b["total"] else 0,
            "avg_latency_ms": round(sum(b["latencies"]) / len(b["latencies"]), 2),
        }

    # Metrics from the collector (captures provider/fallback info)
    collector = metrics.summary()
    llm_stats = collector.get("llm", {})

    report = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(passed / total, 4) if total else 0,
        "type_accuracy": round(type_correct / total, 4) if total else 0,
        "latency": {
            "avg_ms": round(avg_lat, 2),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
        },
        "fallback": {
            "total_llm_calls": llm_stats.get("total_calls", 0),
            "fallback_count": llm_stats.get("fallback_count", 0),
            "fallback_rate": llm_stats.get("fallback_rate", 0),
            "provider_breakdown": llm_stats.get("provider_breakdown", {}),
        },
        "execution": {
            "tested": len(exec_tested),
            "passed": exec_passed,
            "success_rate": round(exec_passed / len(exec_tested), 4) if exec_tested else None,
        },
        "per_type": per_type,
        "details": results,
    }

    # ---- print summary ----
    if verbose:
        _print_summary(report)

    return report


def _print_summary(report: dict) -> None:
    print(f"\n{'=' * 72}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total cases        : {report['total_cases']}")
    print(f"  Passed             : {report['passed']}")
    print(f"  Failed             : {report['failed']}")
    print(f"  Overall accuracy   : {report['accuracy'] * 100:.1f}%")
    print(f"  Type accuracy      : {report['type_accuracy'] * 100:.1f}%")
    print(f"  Avg latency        : {report['latency']['avg_ms']:.1f} ms")
    print(f"  P50 latency        : {report['latency']['p50_ms']:.1f} ms")
    print(f"  P95 latency        : {report['latency']['p95_ms']:.1f} ms")
    print(f"  P99 latency        : {report['latency']['p99_ms']:.1f} ms")
    fb = report["fallback"]
    print(f"  Fallback rate      : {fb['fallback_rate'] * 100:.1f}%")
    print(f"  Provider breakdown : {fb['provider_breakdown']}")
    ex = report["execution"]
    if ex["tested"]:
        print(f"  Exec tested        : {ex['tested']}")
        print(f"  Exec success rate  : {ex['success_rate'] * 100:.1f}%")
    else:
        print(f"  Exec testing       : skipped (use --execute)")
    print(f"\n  Per-type breakdown:")
    for t, s in sorted(report["per_type"].items()):
        print(
            f"    {t:25s}  {s['correct']}/{s['total']}  "
            f"acc={s['accuracy'] * 100:.0f}%  avg={s['avg_latency_ms']:.0f}ms"
        )
    print(f"{'=' * 72}\n")


# --------------- CLI ---------------


def _load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset must be a non-empty JSON array, got {type(data)}")
    # Validate minimal schema
    for i, item in enumerate(data):
        if "command" not in item or "expected" not in item:
            raise ValueError(
                f"Dataset entry {i} missing 'command' or 'expected' key"
            )
    return data


def main():
    logging.basicConfig(level=logging.WARNING)

    ap = argparse.ArgumentParser(
        description="Sara Evaluation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=_DEFAULT_DATASET,
        help="Path to JSON dataset file (default: tests/command_dataset.json)",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Also execute parsed actions against a temporary Excel file",
    )
    ap.add_argument("--report", type=str, help="Save full JSON report to this path")
    ap.add_argument("--csv", type=str, help="Save per-case CSV to this path")
    ap.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = ap.parse_args()

    dataset = _load_dataset(args.dataset)
    report = run_benchmark(dataset, execute=args.execute, verbose=not args.quiet)

    # ---- persist outputs ----
    if args.report:
        clean = dict(report)
        # Strip bulky parsed_action / exec_result from JSON for readability
        clean["details"] = [
            {k: v for k, v in d.items() if k not in ("parsed_action", "exec_result")}
            for d in report["details"]
        ]
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        print(f"  Report saved -> {args.report}")

    if args.csv:
        fieldnames = [
            "command",
            "expected_type",
            "actual_type",
            "type_match",
            "fields_pass",
            "overall_pass",
            "parse_latency_ms",
            "exec_success",
            "exec_latency_ms",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(report["details"])
        print(f"  CSV saved    -> {args.csv}")


if __name__ == "__main__":
    main()
