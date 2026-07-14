from __future__ import annotations

from github_agent_loop.config import AgentConfig
from github_agent_loop.graph import build_graph, initial_state, render_last_assistant_message
from github_agent_loop.main import build_hunt_prompt, format_tooling_summary
class _FakeClient:
    def complete(self, messages, tools):
        return {
            "role": "assistant",
            "content": "finished",
            "used_tools": ["init_fight_analysis", "get_elo_market_signal"],
        }


def test_graph_runs_single_sdk_turn(tmp_path):
    config = AgentConfig(cwd=tmp_path, repository_root=tmp_path)
    graph = build_graph(config=config, client=_FakeClient())
    state = initial_state(config.system_prompt)
    state["messages"].append({"role": "user", "content": "run the hunt"})

    updated_state = graph.invoke(state)

    assert render_last_assistant_message(updated_state) == "finished"
    assert updated_state["used_tools"] == ["init_fight_analysis", "get_elo_market_signal"]


def test_build_hunt_prompt_includes_fight_inputs():
    prompt = build_hunt_prompt(
        fighter1="Alex Perez",
        fighter2="Su Mudaerji",
        fight_date="2026-05-30",
        fighter1_odds=-150,
        fighter2_odds=130,
    )

    assert "fighter1: Alex Perez" in prompt
    assert "fighter2: Su Mudaerji" in prompt
    assert "fight_date: 2026-05-30" in prompt
    assert "fighter1_odds: -150" in prompt
    assert "fighter2_odds: 130" in prompt


def test_agent_config_builds_stdio_mcp_server(tmp_path):
    config = AgentConfig(
        cwd=tmp_path,
        repository_root=tmp_path,
        mcp_server_name="ufc-context-analysis",
        mcp_command=".venv/bin/python",
        mcp_args=("mcp_server/ufc_context_server.py",),
        mcp_cwd=tmp_path,
    )

    servers = config.mcp_servers

    assert "ufc-context-analysis" in servers
    server = servers["ufc-context-analysis"]
    assert server["type"] == "stdio"
    assert server["command"] == ".venv/bin/python"
    assert server["args"] == ["mcp_server/ufc_context_server.py"]
    assert server["cwd"] == str(tmp_path)
    assert server["tools"] == ["*"]


def test_format_tooling_summary_mentions_runtime_and_tools(tmp_path):
    config = AgentConfig(
        cwd=tmp_path,
        repository_root=tmp_path,
        mcp_server_name="ufc-context-analysis",
        mcp_command=".venv/bin/python",
        mcp_args=("mcp_server/ufc_context_server.py",),
        mcp_cwd=tmp_path,
    )

    summary = format_tooling_summary(
        config,
        ["init_fight_analysis", "get_elo_market_signal"],
    )

    assert "LangGraph orchestration" in summary
    assert "authenticated local Copilot CLI" in summary
    assert "ufc-context-analysis" in summary
    assert ".venv/bin/python mcp_server/ufc_context_server.py" in summary
    assert "init_fight_analysis, get_elo_market_signal" in summary
