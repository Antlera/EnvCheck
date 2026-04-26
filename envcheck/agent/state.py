"""
EnvPilot Agent State — Typed state definition for the LangGraph state graph.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class EnvPilotState(TypedDict):
    """Complete state for the EnvPilot agent workflow.

    Each field corresponds to data flowing between the 5 phases.
    """

    # Input
    task_description: str

    # LLM conversation history
    messages: Annotated[list, add_messages]

    # Phase 1: Analysis
    identified_packages: list[str]
    uncertainty_score: int  # 0-100

    # Phase 2: Environment Probing
    env_path: str
    env_info: dict[str, Any]

    # Phase 3: Knowledge Alignment
    kb_results: list[dict[str, Any]]
    kb_has_gaps: bool
    web_results: list[dict[str, Any]]
    kb_updates: list[dict[str, Any]]

    # Phase 4: Preflight
    preflight_code: str
    preflight_result: dict[str, Any]
    preflight_attempts: int

    # Phase 5: Generation
    final_code: str

    # Metadata
    phase: str
    error: str
