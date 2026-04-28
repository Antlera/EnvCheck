"""Run the benchmark eval: for each verified candidate, run both
  - baseline: a single LLM call given task + pip-freeze of bad_env
  - envpilot: the full LangGraph pipeline against bad_env
and score each generated solution by running it + test in bad_env.

Reports per-case rows and aggregate metrics matching the project goals:

  Effectiveness: first-pass success / final success / crash rate
  Efficiency:    latency / tokens / tool calls / overhead vs. repair savings

Usage:
  export GOOGLE_API_KEY=...
  uv run python benchmark/run_eval.py                  # all cases, both modes
  uv run python benchmark/run_eval.py --case manual_006
  uv run python benchmark/run_eval.py --first 5
  uv run python benchmark/run_eval.py --mode baseline  # baseline only
  uv run python benchmark/run_eval.py --mode envpilot  # envpilot only
  uv run python benchmark/run_eval.py --n 3            # repeat each case N times for stability

Output:
  benchmark/eval_results.json      — per-case rows
  benchmark/eval_summary.json      — aggregate metrics
  printed table on stdout
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Allow running as `python benchmark/run_eval.py`
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for envcheck import

from runner_utils import (
    BENCHMARK_DIR,
    build_env,
    compose_test_script,
    load_candidates,
    run_in_env,
    venv_python,
)


RESULTS_PATH = BENCHMARK_DIR / "eval_results.json"
SUMMARY_PATH = BENCHMARK_DIR / "eval_summary.json"


@dataclass
class RunRecord:
    case_id: str
    task_id: str
    rule_label: str
    library_under_test: str
    kind: str
    mode: str  # "baseline" | "envpilot"
    repeat: int  # which repeat-index for stability sampling

    # Generation outcome
    final_code: str
    generation_error: str  # empty if generation succeeded
    duration_s: float

    # Score (running final_code + test in bad_env)
    test_passed: bool
    test_exit_code: int
    test_stderr_tail: str
    test_crashed: bool  # exception, not just assertion failure

    # EnvPilot-only fields (0/empty for baseline)
    preflight_attempts: int = 0
    web_search_called: bool = False
    kb_updates_count: int = 0
    llm_calls: int = 0
    web_search_calls: int = 0
    preflight_runs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # First-pass success: for envpilot, preflight_attempts <= 1 + test passed.
    # For baseline, equals final success (no retries).
    first_pass_success: bool = False


# --------- Baseline: single LLM call ----------

def _pip_freeze(env_path: Path) -> str:
    py = str(venv_python(env_path))
    try:
        r = subprocess.run(
            [py, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout
    except Exception as e:
        return f"# pip freeze failed: {e}"


BASELINE_PROMPT = """You are given a Python coding task and the exact list of
package versions installed in the target environment. Generate the function
body that satisfies the task. The function signature is already provided —
output ONLY the function body (the lines that go after the `def` line),
correctly indented to 4 spaces. Use APIs that are available in the listed
package versions.

# Task
{instruct_prompt}

# Function signature (already declared, don't repeat the def line)
{code_prompt}

# Installed packages in target environment
{pip_freeze}

# Output format
Return only the function body code. Do not include the def line, do not
include markdown fences, do not include any commentary."""


def run_baseline(case: dict, bad_env_path: Path) -> tuple[str, str, dict]:
    """Single LLM call. Returns (final_code, error_str, metrics)."""
    from envcheck.agent.nodes import _get_llm, get_metrics, reset_metrics
    from langchain_core.messages import HumanMessage, SystemMessage

    reset_metrics()
    pip_freeze = _pip_freeze(bad_env_path)
    prompt = BASELINE_PROMPT.format(
        instruct_prompt=case["instruct_prompt"],
        code_prompt=case["code_prompt"],
        pip_freeze=pip_freeze,
    )

    try:
        response = _get_llm().invoke([
            SystemMessage(content="You are an expert Python developer."),
            HumanMessage(content=prompt),
        ])
        text = response.content
        if isinstance(text, list):
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in text
            )
        # Strip markdown fences if present
        code = text.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```")
                             else lines[1:])
        # Track tokens
        from envcheck.agent.nodes import _metrics  # noqa
        _metrics["llm_calls"] += 1
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            _metrics["input_tokens"] += int(usage.get("input_tokens") or 0)
            _metrics["output_tokens"] += int(usage.get("output_tokens") or 0)
            _metrics["total_tokens"] += int(usage.get("total_tokens") or 0)
        return code, "", get_metrics()
    except Exception as e:
        return "", f"{type(e).__name__}: {e}", get_metrics()


# --------- EnvPilot: full LangGraph pipeline ----------

def run_envpilot(case: dict, bad_env_path: Path) -> tuple[str, str, dict, dict]:
    """Run EnvPilot. Returns (final_code, error_str, metrics, state_snapshot)."""
    from envcheck.agent.graph import build_graph, get_default_initial_state
    from envcheck.agent.nodes import get_metrics, reset_metrics

    reset_metrics()
    try:
        app = build_graph()
        state = app.invoke(get_default_initial_state(
            task_description=case["instruct_prompt"],
            env_path=str(bad_env_path),
        ))
        final_code = state.get("final_code", "")
        snap = {
            "preflight_attempts": state.get("preflight_attempts", 0),
            "web_search_called": bool(state.get("web_results")),
            "kb_updates_count": len(state.get("kb_updates") or []),
            "phase_at_end": state.get("phase", ""),
        }
        return final_code, "", get_metrics(), snap
    except Exception as e:
        return "", f"{type(e).__name__}: {e}", get_metrics(), {}


# --------- Scoring ----------

def score_code(case: dict, code: str, bad_env_path: Path,
               timeout_s: int = 90) -> tuple[bool, int, str, bool]:
    """Run code+test in bad_env. Returns (passed, exit_code, stderr_tail, crashed)."""
    if not code.strip():
        return False, -1, "<no code generated>", True

    # If model emitted full function (with def line), strip it.
    body = code
    body_stripped = body.lstrip("\n")
    if body_stripped.startswith("def "):
        # Skip first line (the def), use remaining as body
        idx = body.find("\n")
        body = body[idx + 1 :] if idx != -1 else ""
    # Ensure ends with newline
    if not body.endswith("\n"):
        body += "\n"

    script = compose_test_script(case, user_code=body)
    run = run_in_env(bad_env_path, script, timeout_s=timeout_s)
    # Crash = test runner itself exited abnormally (e.g. import error before
    # any test ran, or a syntax error). assertion failures yield exit code 1
    # too, but unittest signals them as FAIL/ERROR in stderr.
    crashed = run.timed_out or "Traceback" in run.stderr and "FAILED" not in run.stderr
    return run.passed, run.exit_code, run.stderr[-600:], crashed


# --------- Orchestration ----------

def run_one(case: dict, mode: str, repeat: int = 0,
            timeout_s: int = 90) -> RunRecord:
    bad_env_path = build_env(
        case["bad_env_pip"],
        python_version=case.get("bad_python"),
    )

    base = dict(
        case_id=case["case_id"],
        task_id=case["task_id"],
        rule_label=case["rule_label"],
        library_under_test=case["library_under_test"],
        kind=case["kind"],
        mode=mode,
        repeat=repeat,
    )

    start = time.time()
    if mode == "baseline":
        code, err, metrics = run_baseline(case, bad_env_path)
        snap = {}
    elif mode == "envpilot":
        code, err, metrics, snap = run_envpilot(case, bad_env_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    duration = time.time() - start

    # Score
    if err:
        passed, exit_code, stderr_tail, crashed = False, -1, err, True
    else:
        passed, exit_code, stderr_tail, crashed = score_code(
            case, code, bad_env_path, timeout_s=timeout_s,
        )

    attempts = snap.get("preflight_attempts", 0)
    first_pass = bool(passed) and (mode == "baseline" or attempts <= 1)

    return RunRecord(
        **base,
        final_code=code,
        generation_error=err,
        duration_s=round(duration, 2),
        test_passed=passed,
        test_exit_code=exit_code,
        test_stderr_tail=stderr_tail,
        test_crashed=crashed,
        preflight_attempts=attempts,
        web_search_called=bool(snap.get("web_search_called")),
        kb_updates_count=int(snap.get("kb_updates_count", 0)),
        llm_calls=int(metrics.get("llm_calls", 0)),
        web_search_calls=int(metrics.get("web_search_calls", 0)),
        preflight_runs=int(metrics.get("preflight_runs", 0)),
        input_tokens=int(metrics.get("input_tokens", 0)),
        output_tokens=int(metrics.get("output_tokens", 0)),
        total_tokens=int(metrics.get("total_tokens", 0)),
        first_pass_success=first_pass,
    )


# --------- Aggregation ----------

def _safe_mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 2) if xs else 0.0


def aggregate(records: list[RunRecord]) -> dict:
    by_mode: dict[str, list[RunRecord]] = {}
    for r in records:
        by_mode.setdefault(r.mode, []).append(r)

    summary: dict = {}
    for mode, rs in by_mode.items():
        n = len(rs)
        passed = [r for r in rs if r.test_passed]
        first_pass = [r for r in rs if r.first_pass_success]
        crashed = [r for r in rs if r.test_crashed]

        summary[mode] = {
            "n_runs": n,
            "n_unique_cases": len({r.case_id for r in rs}),

            # Effectiveness
            "first_pass_success_rate": round(len(first_pass) / n, 3) if n else 0,
            "final_success_rate":      round(len(passed) / n, 3) if n else 0,
            "crash_rate":              round(len(crashed) / n, 3) if n else 0,

            # Efficiency (means)
            "mean_duration_s":   _safe_mean([r.duration_s for r in rs]),
            "mean_total_tokens": _safe_mean([r.total_tokens for r in rs]),
            "mean_llm_calls":    _safe_mean([r.llm_calls for r in rs]),
            "mean_web_search":   _safe_mean([r.web_search_calls for r in rs]),
            "mean_preflight":    _safe_mean([r.preflight_runs for r in rs]),
            "mean_attempts":     _safe_mean([r.preflight_attempts for r in rs]),
        }

    # Overhead vs. repair savings — only meaningful with both modes
    if "baseline" in by_mode and "envpilot" in by_mode:
        b = summary["baseline"]
        e = summary["envpilot"]
        token_overhead = e["mean_total_tokens"] - b["mean_total_tokens"]
        success_lift = e["final_success_rate"] - b["final_success_rate"]
        summary["delta"] = {
            "token_overhead":      round(token_overhead, 1),
            "tokens_per_extra_pass": (round(token_overhead / success_lift, 1)
                                       if success_lift > 0 else None),
            "success_rate_lift":   round(success_lift, 3),
            "duration_overhead_s": round(e["mean_duration_s"] - b["mean_duration_s"], 2),
        }

    return summary


# --------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="Run only one case_id")
    ap.add_argument("--first", type=int, help="Only first N cases")
    ap.add_argument("--mode", choices=["both", "baseline", "envpilot"],
                    default="both")
    ap.add_argument("--n", type=int, default=1,
                    help="Repeat each (case, mode) N times (LLM is non-deterministic)")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--only-verified", action="store_true", default=True,
                    help="Skip cases where verified=false (default true)")
    args = ap.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        return 2

    cases = load_candidates()
    if args.only_verified:
        cases = [c for c in cases if c.get("verified")]
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]
        if not cases:
            print(f"No verified case_id={args.case}", file=sys.stderr)
            return 1
    if args.first:
        cases = cases[: args.first]

    modes = ["baseline", "envpilot"] if args.mode == "both" else [args.mode]

    print(f"Running {len(cases)} cases × {len(modes)} modes × n={args.n} = "
          f"{len(cases) * len(modes) * args.n} runs\n")

    records: list[RunRecord] = []
    for case in cases:
        for mode in modes:
            for rep in range(args.n):
                tag = f"{case['case_id']}/{mode}/rep{rep}"
                print(f"--- {tag} ---")
                rec = run_one(case, mode, repeat=rep, timeout_s=args.timeout)
                records.append(rec)
                print(f"  passed={rec.test_passed} first_pass={rec.first_pass_success} "
                      f"duration={rec.duration_s}s tokens={rec.total_tokens} "
                      f"llm_calls={rec.llm_calls}")

    # Save
    RESULTS_PATH.write_text(json.dumps([asdict(r) for r in records], indent=2))
    summary = aggregate(records)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nFull results: {RESULTS_PATH}")
    print(f"Summary:      {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
