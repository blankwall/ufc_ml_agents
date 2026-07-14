from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from github_agent_loop.client import GitHubModelsClient
from github_agent_loop.config import AgentConfig


class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], add_messages]
    used_tools: list[str]


def _message_content(message: Any) -> str:
    if hasattr(message, "content"):
        return getattr(message, "content") or ""
    return message.get("content") or ""


def build_graph(
    *,
    config: AgentConfig,
    client: GitHubModelsClient,
    tools: list[dict[str, Any]] | None = None,
):
    del config, tools

    def run_model(state: AgentState) -> dict[str, Any]:
        assistant_message = client.complete(state["messages"], [])
        return {
            "messages": [assistant_message],
            "used_tools": assistant_message.get("used_tools", []),
        }

    graph = StateGraph(AgentState)
    graph.add_node("model", run_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def initial_state(system_prompt: str) -> AgentState:
    return {
        "messages": [{"role": "system", "content": system_prompt}],
        "used_tools": [],
    }


def render_last_assistant_message(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        role = getattr(message, "type", None) or message.get("role")
        if role == "ai" or role == "assistant":
            content = _message_content(message)
            return content.strip() or json.dumps(message, indent=2)
    return ""
