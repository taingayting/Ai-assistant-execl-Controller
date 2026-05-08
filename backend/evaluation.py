"""Evaluation framework for Sara.

Measures:
  - Intent parsing accuracy  (LLM output matches expected action)
  - Command execution success rate
  - LLM latency per provider
  - Fallback frequency  (Groq → Ollama → heuristic)

Runs a benchmark from a JSON test dataset and produces a report
suitable for thesis/paper inclusion.

Usage::

    python evaluation.py                     # default dataset
    python evaluation.py --dataset my.json   # custom dataset
    python evaluation.py --report report.json --csv report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env):
    load_dotenv(_env, override=True)

from llm_router import route_llm
from excel_controller import execute_action, create_excel
from metrics import metrics

log = logging.getLogger('evaluation')

# --------------- Default benchmark dataset ---------------

DEFAULT_DATASET: list[dict] = [
    # --- create_excel ---
    {"input": "create excel with headers Name, Score",
     "expected_type": "create_excel", "expected_fields": {"headers": ["Name", "Score"]}},
    {"input": "make a new spreadsheet with columns Item, Quantity",
     "expected_type": "create_excel", "expected_fields": {"headers": ["Item", "Quantity"]}},
    {"input": "create a new file with headers Name, Age, Grade",
     "expected_type": "create_excel"},
    # --- add_row ---
    {"input": "add row Tom 90",
     "expected_type": "add_row", "expected_fields": {"values": ["Tom", 90]}},
    {"input": "insert row Alice 85",
     "expected_type": "add_row"},
    {"input": "add a new row with values Laptop, 5, 999",
     "expected_type": "add_row"},
    # --- delete_row ---
    {"input": "delete row 2",
     "expected_type": "delete_row", "expected_fields": {"index": 2}},
    {"input": "remove the 3rd row",
     "expected_type": "delete_row"},
    # --- write_cell ---
    {"input": "write 100 to B3",
     "expected_type": "write_cell", "expected_fields": {"cell": "B3", "value": 100}},
    {"input": "put hello in cell A1",
     "expected_type": "write_cell"},
    # --- add_column ---
    {"input": "add column called Revenue",
     "expected_type": "add_column", "expected_fields": {"name": "Revenue"}},
    # --- delete_column ---
    {"input": "delete column B",
     "expected_type": "delete_column", "expected_fields": {"column": "B"}},
    # --- calculate ---
    {"input": "calculate sum of row 1",
     "expected_type": "calculate_row_sum", "expected_fields": {"index": 1}},
    # --- multi-step ---
    {"input": "create an excel sheet for student scores and add Tom with 90 and Alex with 85",
     "expected_type": "steps", "expected_step_count": 3},
    {"input": "make an attendance table and add Alice present and Bob absent",
     "expected_type": "steps"},
    # --- save ---
    {"input": "save the file",
     "expected_type": "save"},
    # --- ambiguous / edge cases ---
    {"input": "add row",
     "expected_type": "unknown", "notes": "incomplete command"},
    {"input": "hello how are you",
     "expected_type": "unknown", "notes": "not an Excel command"},
    {"input": "Create a score sheet and add Tom 90",
     "expected_type": "steps"},
    {"input": "make an expense tracker and add lunch 25 and dinner 40",
     "expected_type": "steps"},
    # --- write_formula (new) ---
    {"input": "write formula SUM(A2:A10) in cell A11",
     "expected_type": "write_formula"},
]


# --------------- Evaluation logic ---------------

def _action_type(result: dict) -> str:
    """Extract the effective action type from a parsed result."""
    if isinstance(result.get('steps'), list) and result['steps']:
        return 'steps'
    return result.get('type', 'unknown')


def _fields_match(result: dict, expected: dict) -> bool:
    """Check if specific expected fields match in the result."""
    for key, val in expected.items():
        actual = result.get(key)
        if isinstance(val, list) and isinstance(actual, list):
            if len(val) != len(actual):
                return False
            for v, a in zip(val, actual):
                if str(v).lower() != str(a).lower():
                    return False
        elif isinstance(val, str) and isinstance(actual, str):
            if val.upper() != actual.upper():
                return False
        elif val != actual:
            return False
    return True


def evaluate_single(case: dict) -> dict:
    """Run one test case and return detailed results."""
    text = case['input']
    expected_type = case.get('expected_type', '')
    expected_fields = case.get('expected_fields', {})
    expected_steps = case.get('expected_step_count')

    t0 = time.perf_counter()
    result = route_llm(text)
    latency_ms = (time.perf_counter() - t0) * 1000

    actual_type = _action_type(result)
    type_match = actual_type == expected_type

    fields_match = True
    if expected_fields:
        fields_match = _fields_match(result, expected_fields)

    steps_match = True
    if expected_steps is not None:
        actual_steps = len(result.get('steps', []))
        steps_match = actual_steps >= expected_steps

    return {
        'input': text,
        'expected_type': expected_type,
        'actual_type': actual_type,
        'type_match': type_match,
        'fields_match': fields_match,
        'steps_match': steps_match,
        'overall_pass': type_match and fields_match and steps_match,
        'latency_ms': round(latency_ms, 2),
        'result': result,
        'notes': case.get('notes', ''),
    }


def run_benchmark(dataset: list[dict] | None = None, verbose: bool = True) -> dict:
    """Execute the full benchmark suite and return a summary report."""
    cases = dataset or DEFAULT_DATASET
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Sara Evaluation Benchmark — {len(cases)} test cases")
        print(f"{'='*70}\n")

    for i, case in enumerate(cases, 1):
        r = evaluate_single(case)
        results.append(r)
        if verbose:
            status = '✓ PASS' if r['overall_pass'] else '✗ FAIL'
            print(f"  [{i:2d}] {status}  {r['latency_ms']:7.1f}ms  "
                  f"expected={r['expected_type']:20s}  got={r['actual_type']:20s}  "
                  f"| {r['input'][:50]}")

    # Aggregate
    total = len(results)
    passed = sum(1 for r in results if r['overall_pass'])
    type_correct = sum(1 for r in results if r['type_match'])
    avg_latency = sum(r['latency_ms'] for r in results) / total if total else 0
    latencies = [r['latency_ms'] for r in results]
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    # Metrics from collector
    collector_summary = metrics.summary()

    report = {
        'total_cases': total,
        'passed': passed,
        'failed': total - passed,
        'accuracy': round(passed / total, 4) if total else 0,
        'type_accuracy': round(type_correct / total, 4) if total else 0,
        'avg_latency_ms': round(avg_latency, 2),
        'p50_latency_ms': round(p50, 2),
        'p95_latency_ms': round(p95, 2),
        'llm_metrics': collector_summary.get('llm', {}),
        'details': results,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"  RESULTS")
        print(f"{'='*70}")
        print(f"  Total cases    : {total}")
        print(f"  Passed         : {passed}")
        print(f"  Failed         : {total - passed}")
        print(f"  Accuracy       : {report['accuracy']*100:.1f}%")
        print(f"  Type accuracy  : {report['type_accuracy']*100:.1f}%")
        print(f"  Avg latency    : {avg_latency:.1f} ms")
        print(f"  P50 latency    : {p50:.1f} ms")
        print(f"  P95 latency    : {p95:.1f} ms")
        fb = collector_summary.get('llm', {})
        print(f"  Fallback rate  : {fb.get('fallback_rate', 0)*100:.1f}%")
        print(f"  Provider usage : {fb.get('provider_breakdown', {})}")
        print(f"{'='*70}\n")

    return report


# --------------- CLI ---------------

def main():
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description='Sara Evaluation Benchmark')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to a JSON benchmark dataset file')
    parser.add_argument('--report', type=str, default=None,
                        help='Save JSON report to this path')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save CSV results to this path')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress console output')
    args = parser.parse_args()

    dataset = None
    if args.dataset:
        with open(args.dataset, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

    report = run_benchmark(dataset=dataset, verbose=not args.quiet)

    if args.report:
        # Strip full result dicts from details for cleaner JSON
        clean = dict(report)
        clean['details'] = [{k: v for k, v in d.items() if k != 'result'}
                            for d in report['details']]
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.report}")

    if args.csv:
        rows = report['details']
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'input', 'expected_type', 'actual_type', 'type_match',
                'fields_match', 'steps_match', 'overall_pass', 'latency_ms', 'notes',
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, '') for k in writer.fieldnames})
        print(f"CSV saved to {args.csv}")


if __name__ == '__main__':
    main()
