from __future__ import annotations

import argparse
from pathlib import Path

from github_agent_loop.client import GitHubModelsClient
from github_agent_loop.config import AgentConfig
from github_agent_loop.graph import build_graph, initial_state, render_last_assistant_message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal LangGraph + Copilot SDK UFC hunt loop.")
    parser.add_argument("--prompt", type=str, help="Single prompt to run. If omitted, starts an interactive loop.")
    parser.add_argument("--fighter1", type=str, help="First fighter name.")
    parser.add_argument("--fighter2", type=str, help="Second fighter name.")
    parser.add_argument("--fight-date", type=str, help="Fight date in YYYY-MM-DD format.")
    parser.add_argument("--fighter1-odds", type=int, help="American odds for fighter1.")
    parser.add_argument("--fighter2-odds", type=int, help="American odds for fighter2.")
    parser.add_argument("--model", type=str, default=None, help="GitHub Models model name.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Repository working directory.")
    parser.add_argument("--cli-url", type=str, default=None, help="Optional host:port for an already running Copilot CLI server.")
    parser.add_argument("--verbose", action="store_true", help="Print tool calls to stderr.")
    return parser


def build_hunt_prompt(
    *,
    fighter1: str,
    fighter2: str,
    fight_date: str | None,
    fighter1_odds: int | None,
    fighter2_odds: int | None,
) -> str:
    lines = [
        "Hunt this UFC matchup with the full mandatory MCP workflow and produce the final decision-support analysis.",
        f"fighter1: {fighter1}",
        f"fighter2: {fighter2}",
    ]
    if fight_date:
        lines.append(f"fight_date: {fight_date}")
    if fighter1_odds is not None:
        lines.append(f"fighter1_odds: {fighter1_odds}")
    if fighter2_odds is not None:
        lines.append(f"fighter2_odds: {fighter2_odds}")
    lines.append(
        "Use every mandatory evidence lane before concluding, and give an exact odds threshold in the final answer."
    )
    return "\n".join(lines)


def format_tooling_summary(config: AgentConfig, used_tools: list[str]) -> str:
    reported_tools = ", ".join(used_tools) if used_tools else "none reported"
    transport = (
        f"Copilot SDK via external CLI server ({config.cli_url})"
        if config.cli_url
        else "Copilot SDK via authenticated local Copilot CLI"
    )
    mcp_command = " ".join([config.mcp_command, *config.mcp_args])
    return "\n".join(
        [
            "Tooling used:",
            f"- Runtime: LangGraph orchestration",
            f"- Model transport: {transport}",
            f"- Analysis tools: custom SDK tools backed by MCP stdio server `{config.mcp_server_name}`",
            f"- MCP server command: {mcp_command}",
            f"- MCP tools invoked: {reported_tools}",
        ]
    )


def run_prompt(graph, state, prompt: str):
    state["messages"].append({"role": "user", "content": prompt})
    updated_state = graph.invoke(state)
    return updated_state


def main() -> int:
    args = build_parser().parse_args()
    config = AgentConfig.from_env(
        cwd=args.cwd,
        model=args.model,
        verbose=args.verbose,
    )
    if args.cli_url:
        config = AgentConfig(
            model=config.model,
            cwd=config.cwd,
            repository_root=config.repository_root,
            cli_url=args.cli_url,
            cli_path=config.cli_path,
            mcp_server_name=config.mcp_server_name,
            mcp_command=config.mcp_command,
            mcp_args=config.mcp_args,
            mcp_cwd=config.mcp_cwd,
            mcp_timeout_ms=config.mcp_timeout_ms,
            request_timeout_seconds=config.request_timeout_seconds,
            max_reminders=config.max_reminders,
            system_prompt=config.system_prompt,
            required_tools=config.required_tools,
            verbose=config.verbose,
        )
    graph = build_graph(
        config=config,
        client=GitHubModelsClient(config),
    )
    state = initial_state(config.system_prompt)

    fight_prompt = None
    if args.fighter1 or args.fighter2:
        if not (args.fighter1 and args.fighter2):
            raise SystemExit("Pass both --fighter1 and --fighter2.")
        fight_prompt = build_hunt_prompt(
            fighter1=args.fighter1,
            fighter2=args.fighter2,
            fight_date=args.fight_date,
            fighter1_odds=args.fighter1_odds,
            fighter2_odds=args.fighter2_odds,
        )

    if args.prompt or fight_prompt:
        state = run_prompt(graph, state, fight_prompt or args.prompt)
        print(render_last_assistant_message(state))
        print()
        print(format_tooling_summary(config, state.get("used_tools", [])))
        return 0

    while True:
        try:
            prompt = input("agent> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0
        state = run_prompt(graph, state, prompt)
        print(render_last_assistant_message(state))
        print()
        print(format_tooling_summary(config, state.get("used_tools", [])))


if __name__ == "__main__":
    raise SystemExit(main())
