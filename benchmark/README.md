# EnvCheck Benchmark

A small benchmark for evaluating EnvPilot's ability to handle Python environments where a target library has a known breaking change — i.e. `canonical_solution` works in one library version and crashes in another due to API removal/rename, and EnvPilot must generate code that runs in the broken environment.

## Files

| File | Purpose |
|---|---|
| `build_candidates.py` | Builds `candidates.json` from BigCodeBench v0.1.4 + `manual_cases.py`. Re-runnable. |
| `manual_cases.py` | 8 hand-written cases in BigCodeBench dict format covering numpy 2.0 / pandas 2.0 / Pillow 10 / flask 2.3 removals. |
| `runner_utils.py` | Shared utilities: build cached `uv` venvs from a pip-spec list, compose code+test scripts, run with timeout. |
| `verify_ground_truth.py` | Verifies each candidate: runs `canonical_solution + test` in both `bad_env` and `good_env`, expects bad to fail with documented `error_type` and good to pass. |
| `candidates.json` | **The benchmark itself.** 21 verified `(task, bad_env, good_env)` cases with full BCB-style fields. Generated but committed for convenience. |
| `bigcodebench_pool.json` | Early 50-task random sample from BCB filtered to target libs. Kept for reference; not used by current build pipeline. |
| `pool_stats.json` | Library distribution of the original BCB filtered pool. Kept for reference. |
| `verification_report.json` | (gitignored) Output of last `verify_ground_truth.py` run. |
| `envs/` | (gitignored) Cached venvs keyed by hash of (Python version, pip spec). |

## Usage

```bash
# 1. Re-generate candidates.json from sources (BCB + manual_cases.py)
uv run --with datasets python benchmark/build_candidates.py

# 2. Verify ground truth: build bad_env + good_env per case, run canonical+test
python benchmark/verify_ground_truth.py             # all cases
python benchmark/verify_ground_truth.py --case manual_006   # one case
python benchmark/verify_ground_truth.py --first 5           # first 5
python benchmark/verify_ground_truth.py --update            # write `verified` field back into candidates.json

# Force rebuild venvs (otherwise cached under benchmark/envs/<hash>/)
python benchmark/verify_ground_truth.py --rebuild
```

The first `verify_ground_truth.py` run takes 5–10 minutes (creates ~10 unique venvs across cases). Subsequent runs are seconds (envs are cached by `(python_version, sorted(pip_spec))` hash, so cases sharing identical envs share one venv).

## Case schema (`candidates.json`)

Each entry is a dict:

```json
{
  "case_id": "bcb_002",
  "task_id": "BigCodeBench/53",
  "libs": ["regex", "pandas", "matplotlib", "seaborn"],
  "library_under_test": "seaborn",
  "bad_version": "0.10.1",
  "good_version": "0.13.2",
  "bad_env_pip":  ["matplotlib==3.2.2", "numpy==1.18.5", "pandas==1.0.5", "regex", "seaborn==0.10.1"],
  "good_env_pip": ["matplotlib==3.8.4", "numpy==1.26.4", "pandas==2.2.2", "regex", "seaborn==0.13.2"],
  "bad_python":  "3.8",
  "good_python": "3.11",
  "error_type": "AttributeError",
  "kind": "introduction",
  "rule_label": "sns_histplot",
  "reason": "sns.histplot added in seaborn 0.11",
  "evidence_line": "    sns.histplot(data=df, x=\"Age\")",
  "note": "",
  "instruct_prompt": "...",
  "code_prompt": "...",
  "canonical_solution": "...",
  "test": "...",
  "entry_point": "task_func",
  "verified": true
}
```

- `kind` — `"introduction"` (canonical uses an API that didn't exist in `bad_version`) or `"removal"` (canonical uses an API that was removed in `bad_version`).
- `bad_env_pip` / `good_env_pip` — full pip-installable lists. Peer libs are pinned to mutually-compatible snapshot versions to avoid (a) a second confounding break and (b) ABI mismatches (e.g. pandas wheel ↔ numpy version).
- `bad_python` / `good_python` — venv Python version. Some intro cases (sns.histplot, sns.displot) need Python 3.8 because their old peers (matplotlib 3.2.2, numpy 1.18.5) lack 3.10+ wheels; everything else uses 3.11.
- `verified` — true if `verify_ground_truth.py` confirmed canonical+test crashes in bad_env and passes in good_env, with the documented `error_type` appearing in stderr.

## Distribution

21 cases:
- **Libraries**: seaborn (12) · numpy (3) · pandas (3) · scikit-learn (1) · Pillow (1) · flask (1)
- **Direction**: introduction (12) · removal (9)
- **Error types**: AttributeError (18) · TypeError (2) · ImportError (1)
- **Source**: BCB directed regex search (13) · hand-written (8)

## Data sources & attribution

- **BigCodeBench** (`bigcode/bigcodebench`, split `v0.1.4`) — Apache License 2.0 — https://huggingface.co/datasets/bigcode/bigcodebench. The `task_id`, `libs`, `instruct_prompt`, `code_prompt`, `canonical_solution`, `test`, and `entry_point` fields of `bcb_*` cases are derived from this dataset. Selection of cases was via regex search for known breaking-change patterns; the dataset content itself is not modified.
- **Manual cases** — hand-written in BigCodeBench dict format following the `Manual/N` task_id convention. Source-of-truth lives in `manual_cases.py`.
- **Breaking-change version pairs** — cross-referenced from official release notes / migration guides of NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn, Pillow, and Flask.
