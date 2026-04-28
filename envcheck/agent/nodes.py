"""
EnvPilot Agent Nodes — Implementation of each phase in the state graph.

Each node function takes the current EnvPilotState and returns a partial
state update. The LLM is called via langchain-google-genai for analysis phases.
"""

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from envcheck.agent.prompts import (
    ANALYSIS_PROMPT,
    ENV_PROBE_ANALYSIS_PROMPT,
    GENERATION_PROMPT,
    KB_ANALYSIS_PROMPT,
    KB_UPDATE_PROMPT,
    PREFLIGHT_PROMPT,
    SYSTEM_PROMPT,
)
from envcheck.agent.state import EnvPilotState
from envcheck.knowledge_base_store import KnowledgeBaseStore
from envcheck.preflight_runner import run_preflight, to_dict as preflight_to_dict
from envcheck.version_detector import get_installed_packages
from envcheck.web_searcher import WebSearcher

logger = logging.getLogger("envpilot")

MAX_PREFLIGHT_ATTEMPTS = 3

# Per-run instrumentation for benchmark eval.
# Reset at the start of each EnvPilot run, then read after invoke().
_metrics: dict[str, int] = {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "web_search_calls": 0,
    "preflight_runs": 0,
    "kb_query_calls": 0,
}


def reset_metrics() -> None:
    """Zero out all instrumentation counters. Call before each pipeline run."""
    for k in _metrics:
        _metrics[k] = 0


def get_metrics() -> dict[str, int]:
    """Snapshot current counters."""
    return dict(_metrics)


def _get_llm(model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, temperature=0, max_tokens=4096)


def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        start = 1
        end = len(lines) - 1
        if lines[-1].strip() == "```":
            cleaned = "\n".join(lines[start:end])
        else:
            cleaned = "\n".join(lines[start:])
    return json.loads(cleaned)


def _llm_call(prompt: str) -> dict:
    """Make an LLM call and parse the JSON response."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)

    # Instrumentation: count call + tokens for benchmark eval.
    _metrics["llm_calls"] += 1
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        _metrics["input_tokens"] += int(usage.get("input_tokens") or 0)
        _metrics["output_tokens"] += int(usage.get("output_tokens") or 0)
        _metrics["total_tokens"] += int(usage.get("total_tokens") or 0)

    text = response.content
    if isinstance(text, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in text
        )
    return _parse_json_response(text)


# ============================================================================
# Phase 1: Analysis & Uncertainty Assessment
# ============================================================================

def analysis_node(state: EnvPilotState) -> dict[str, Any]:
    """Analyze the task, identify packages, and assign uncertainty score."""
    logger.info("[Phase 1] Analyzing task and assessing uncertainty...")

    prompt = ANALYSIS_PROMPT.format(task_description=state["task_description"])

    try:
        result = _llm_call(prompt)
        return {
            "identified_packages": result.get("identified_packages", []),
            "uncertainty_score": result.get("uncertainty_score", 50),
            "phase": "analysis_complete",
            "messages": [HumanMessage(content=f"[Analysis] {json.dumps(result, indent=2)}")],
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "identified_packages": [],
            "uncertainty_score": 80,
            "phase": "analysis_error",
            "error": str(e),
            "messages": [HumanMessage(content=f"[Analysis Error] {e}")],
        }


# ============================================================================
# Phase 2: Environment Probing
# ============================================================================

def env_probe_node(state: EnvPilotState) -> dict[str, Any]:
    """Probe the local environment for installed package versions."""
    logger.info("[Phase 2] Probing environment...")

    env_path = state.get("env_path", "")
    packages = state.get("identified_packages", [])

    env_info: dict[str, Any] = {"packages": {}, "python": "", "probed": True}

    if env_path:
        from pathlib import Path
        installed = get_installed_packages(env_path)
        python_bin = Path(env_path) / "bin" / "python"
        if python_bin.exists():
            import subprocess
            try:
                ver = subprocess.run(
                    [str(python_bin), "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                env_info["python"] = ver.stdout.strip()
            except Exception:
                env_info["python"] = "unknown"
    else:
        import sys
        from pathlib import Path
        installed = get_installed_packages(Path(sys.prefix))
        import platform
        env_info["python"] = platform.python_version()

    for pkg_name in packages:
        normalized = pkg_name.lower()
        if normalized in installed:
            env_info["packages"][pkg_name] = {
                "installed": True,
                "version": installed[normalized].version,
            }
        else:
            env_info["packages"][pkg_name] = {
                "installed": False,
                "version": None,
            }

    # LLM analysis of env results
    prompt = ENV_PROBE_ANALYSIS_PROMPT.format(env_info=json.dumps(env_info, indent=2))
    try:
        llm_result = _llm_call(prompt)
        updated_uncertainty = llm_result.get(
            "updated_uncertainty_score", state.get("uncertainty_score", 50)
        )
    except Exception:
        updated_uncertainty = state.get("uncertainty_score", 50)

    return {
        "env_info": env_info,
        "uncertainty_score": updated_uncertainty,
        "phase": "env_probed",
        "messages": [HumanMessage(content=f"[Env Probe] {json.dumps(env_info, indent=2)}")],
    }


# ============================================================================
# Phase 3: Knowledge Alignment
# ============================================================================

def kb_query_node(state: EnvPilotState) -> dict[str, Any]:
    """Query the knowledge base for relevant breaking change rules."""
    logger.info("[Phase 3] Querying knowledge base...")
    _metrics["kb_query_calls"] += 1

    store = KnowledgeBaseStore()
    packages = state.get("identified_packages", [])
    all_results: list[dict] = []

    for pkg in packages:
        rules = store.query(library=pkg)
        all_results.extend(store.to_dict_list(rules))

    env_info = state.get("env_info", {})

    prompt = KB_ANALYSIS_PROMPT.format(
        kb_results=json.dumps(all_results, indent=2),
        env_info=json.dumps(env_info, indent=2),
    )

    kb_has_gaps = False
    try:
        llm_result = _llm_call(prompt)
        kb_has_gaps = llm_result.get("kb_has_gaps", False)
        updated_uncertainty = llm_result.get(
            "updated_uncertainty_score", state.get("uncertainty_score", 50)
        )
    except Exception:
        kb_has_gaps = len(all_results) == 0
        updated_uncertainty = state.get("uncertainty_score", 50)

    store.close()

    return {
        "kb_results": all_results,
        "kb_has_gaps": kb_has_gaps,
        "uncertainty_score": updated_uncertainty,
        "phase": "kb_queried",
        "messages": [HumanMessage(
            content=f"[KB Query] Found {len(all_results)} rules. Gaps: {kb_has_gaps}"
        )],
    }


def web_search_node(state: EnvPilotState) -> dict[str, Any]:
    """Search the web for missing API documentation and breaking changes."""
    logger.info("[Phase 3b] Searching web for missing information...")
    _metrics["web_search_calls"] += 1

    searcher = WebSearcher()
    packages = state.get("identified_packages", [])
    all_results: list[dict] = []

    for pkg in packages:
        env_pkg_info = state.get("env_info", {}).get("packages", {}).get(pkg, {})
        version = env_pkg_info.get("version", "")
        query = f"breaking changes migration guide"
        if version:
            query = f"version {version} {query}"

        results = searcher.search_api_docs(pkg, query, max_results=3)
        all_results.extend(searcher.to_dict_list(results))

    return {
        "web_results": all_results,
        "phase": "web_searched",
        "messages": [HumanMessage(
            content=f"[Web Search] Found {len(all_results)} results"
        )],
    }


def kb_update_node(state: EnvPilotState) -> dict[str, Any]:
    """Extract and upsert new rules from web search results into the KB."""
    logger.info("[Phase 3c] Updating knowledge base with web findings...")

    web_results = state.get("web_results", [])
    if not web_results:
        return {
            "kb_updates": [],
            "phase": "kb_updated",
            "messages": [HumanMessage(content="[KB Update] No web results to process")],
        }

    prompt = KB_UPDATE_PROMPT.format(web_results=json.dumps(web_results, indent=2))

    kb_updates: list[dict] = []
    try:
        llm_result = _llm_call(prompt)
        rules_to_upsert = llm_result.get("rules_to_upsert", [])

        if rules_to_upsert:
            from envcheck.knowledge_base import BreakingChangeRule, PatternType, Severity
            store = KnowledgeBaseStore()
            for rule_dict in rules_to_upsert:
                try:
                    br = BreakingChangeRule(
                        rule_id=rule_dict["rule_id"],
                        library=rule_dict["library"],
                        removed_in=rule_dict["removed_in"],
                        pattern_type=PatternType(rule_dict["pattern_type"]),
                        module_path=rule_dict["module_path"],
                        symbol=rule_dict["symbol"],
                        old_api=rule_dict["old_api"],
                        new_api=rule_dict["new_api"],
                        error_type=rule_dict["error_type"],
                        description=rule_dict["description"],
                        severity=Severity(rule_dict.get("severity", "error")),
                    )
                    store.upsert(br, source="web_search")
                    kb_updates.append(rule_dict)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid rule: {e}")
            store.close()
    except Exception as e:
        logger.warning(f"KB update LLM call failed: {e}")

    return {
        "kb_updates": kb_updates,
        "phase": "kb_updated",
        "messages": [HumanMessage(
            content=f"[KB Update] Upserted {len(kb_updates)} new rules"
        )],
    }


# ============================================================================
# Phase 4: Pre-flight Verification
# ============================================================================

def preflight_node(state: EnvPilotState) -> dict[str, Any]:
    """Generate and run a preflight smoke test."""
    logger.info("[Phase 4] Running preflight verification...")
    _metrics["preflight_runs"] += 1

    attempts = state.get("preflight_attempts", 0) + 1
    env_path = state.get("env_path", "")

    if not env_path:
        return {
            "preflight_result": {"success": True, "note": "No env_path, skipping preflight"},
            "preflight_attempts": attempts,
            "phase": "preflight_passed",
            "messages": [HumanMessage(content="[Preflight] Skipped (no env_path)")],
        }

    prompt = PREFLIGHT_PROMPT.format(
        env_info=json.dumps(state.get("env_info", {}), indent=2),
        kb_results=json.dumps(state.get("kb_results", []), indent=2),
        web_results=json.dumps(state.get("web_results", []), indent=2),
        task_description=state["task_description"],
    )

    try:
        llm_result = _llm_call(prompt)
        code = llm_result.get("preflight_code", "")
    except Exception as e:
        return {
            "preflight_result": {"success": False, "error": str(e)},
            "preflight_attempts": attempts,
            "phase": "preflight_failed",
            "error": str(e),
            "messages": [HumanMessage(content=f"[Preflight] LLM failed: {e}")],
        }

    result = run_preflight(code, env_path)
    result_dict = preflight_to_dict(result)

    phase = "preflight_passed" if result.success else "preflight_failed"

    return {
        "preflight_code": code,
        "preflight_result": result_dict,
        "preflight_attempts": attempts,
        "phase": phase,
        "messages": [HumanMessage(
            content=f"[Preflight] {'PASSED' if result.success else 'FAILED'}: {result_dict}"
        )],
    }


# ============================================================================
# Phase 5: Final One-Pass Generation
# ============================================================================

def generation_node(state: EnvPilotState) -> dict[str, Any]:
    """Generate the final code with full environment context."""
    logger.info("[Phase 5] Generating final code...")

    prompt = GENERATION_PROMPT.format(
        env_info=json.dumps(state.get("env_info", {}), indent=2),
        kb_results=json.dumps(state.get("kb_results", []), indent=2),
        preflight_result=json.dumps(state.get("preflight_result", {}), indent=2),
        task_description=state["task_description"],
    )

    try:
        llm_result = _llm_call(prompt)
        final_code = llm_result.get("final_code", "")
        notes = llm_result.get("notes", "")
    except Exception as e:
        return {
            "final_code": "",
            "phase": "generation_error",
            "error": str(e),
            "messages": [HumanMessage(content=f"[Generation Error] {e}")],
        }

    return {
        "final_code": final_code,
        "phase": "complete",
        "messages": [HumanMessage(
            content=f"[Generation Complete]\nNotes: {notes}\n\nCode:\n```python\n{final_code}\n```"
        )],
    }


# ============================================================================
# Routing functions for conditional edges
# ============================================================================

def route_after_kb_query(state: EnvPilotState) -> str:
    """Decide whether to web search or go straight to preflight."""
    uncertainty = state.get("uncertainty_score", 50)
    kb_has_gaps = state.get("kb_has_gaps", False)

    if uncertainty > 20 or kb_has_gaps:
        return "web_search"
    return "preflight"


def route_after_preflight(state: EnvPilotState) -> str:
    """Decide whether to retry analysis or proceed to generation."""
    result = state.get("preflight_result", {})
    attempts = state.get("preflight_attempts", 0)

    if result.get("success", False):
        return "generation"

    if attempts >= MAX_PREFLIGHT_ATTEMPTS:
        logger.warning(f"Max preflight attempts ({MAX_PREFLIGHT_ATTEMPTS}) reached, proceeding anyway")
        return "generation"

    return "analysis"
