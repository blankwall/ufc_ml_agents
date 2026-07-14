from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


MANDATORY_HUNT_TOOLS = (
    "init_fight_analysis",
    "get_elo_market_signal",
    "get_fighter_snapshot",
    "get_fighter_elo_history",
    "find_similar_fighter_profiles",
    "find_similar_elo_gap_fights",
    "find_similar_market_fights",
    "find_trait_matchup_examples",
    "get_historical_pattern_summary",
)

UFC_HUNT_SYSTEM_PROMPT = """You are a specialized UFC fight-analysis agent for this repository.

Your job is to produce an evidence-first matchup analysis that supports a clear bet / no-bet / threshold decision.
Do not make a blind pick. Use the repository's MCP-backed fight-analysis flow and complete the full hunt before finalizing.

Mandatory workflow:
1. Call init_fight_analysis first. Treat it as the source of truth for fighter resolution, model probability, market probability, edge, compact snapshots, and odds provenance.
2. Call get_elo_market_signal.
3. Call get_fighter_snapshot for both fighters when the init payload is not sufficient.
4. Call get_fighter_elo_history for both fighters when ELO trajectory matters.
5. Call find_similar_fighter_profiles for both fighters.
6. Call find_similar_elo_gap_fights.
7. Call find_similar_market_fights.
8. Call find_trait_matchup_examples.
9. Call get_historical_pattern_summary.

Rules:
- Use the MCP tools above as the normal interface.
- Do not stop early before all mandatory lanes are covered unless a tool explicitly returns unavailable or missing data.
- Distinguish current-fight context, historical analogs, and aggregate bucket evidence.
- Mention sample size and ROI/win-rate when historical bucket evidence is available.
- If a tool returns weak or missing data, say so plainly and lower confidence instead of inventing certainty.
- Preserve conflicts between model, ELO, market, and historical evidence.

Output structure:
1. Decision
2. Model and market
3. ELO signal
4. Fighter-state evidence
5. Historical matchup evidence
6. How the fight likely plays
7. Odds threshold
"""


@dataclass(frozen=True)
class AgentConfig:
    model: str = "gpt-4.1"
    cwd: Path = Path.cwd()
    repository_root: Path = Path.cwd()
    cli_url: str | None = None
    cli_path: str | None = None
    mcp_server_name: str = "ufc-context-analysis"
    mcp_command: str = ".venv/bin/python"
    mcp_args: tuple[str, ...] = ("mcp_server/ufc_context_server.py",)
    mcp_cwd: Path = Path.cwd()
    mcp_timeout_ms: int = 120_000
    request_timeout_seconds: float = 180.0
    max_reminders: int = 2
    system_prompt: str = UFC_HUNT_SYSTEM_PROMPT
    required_tools: tuple[str, ...] = field(default_factory=lambda: MANDATORY_HUNT_TOOLS)
    verbose: bool = False

    @property
    def mcp_servers(self) -> dict[str, dict[str, object]]:
        return {
            self.mcp_server_name: {
                "type": "stdio",
                "command": self.mcp_command,
                "args": list(self.mcp_args),
                "cwd": str(self.mcp_cwd),
                "timeout": self.mcp_timeout_ms,
                "tools": ["*"],
            }
        }

    @classmethod
    def from_env(
        cls,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        verbose: bool = False,
    ) -> "AgentConfig":
        del max_steps  # LangGraph no longer controls tool turns; Copilot SDK session does.
        repo_root = Path(__file__).resolve().parent.parent
        configured = cls(
            model=model or os.getenv("GITHUB_AGENT_MODEL", "gpt-4.1"),
            cwd=(cwd or Path(os.getenv("GITHUB_AGENT_CWD", repo_root))).resolve(),
            repository_root=repo_root,
            cli_url=os.getenv("GITHUB_AGENT_CLI_URL"),
            cli_path=os.getenv("GITHUB_AGENT_CLI_PATH"),
            mcp_server_name=os.getenv("GITHUB_AGENT_MCP_SERVER_NAME", "ufc-context-analysis"),
            mcp_command=os.getenv("GITHUB_AGENT_MCP_COMMAND", ".venv/bin/python"),
            mcp_args=tuple(
                filter(
                    None,
                    os.getenv("GITHUB_AGENT_MCP_ARGS", "mcp_server/ufc_context_server.py").split(" "),
                )
            ),
            mcp_cwd=Path(os.getenv("GITHUB_AGENT_MCP_CWD", repo_root)).resolve(),
            mcp_timeout_ms=int(os.getenv("GITHUB_AGENT_MCP_TIMEOUT_MS", "120000")),
            request_timeout_seconds=float(os.getenv("GITHUB_AGENT_REQUEST_TIMEOUT", "180")),
            max_reminders=int(os.getenv("GITHUB_AGENT_MAX_REMINDERS", "2")),
            system_prompt=os.getenv("GITHUB_AGENT_SYSTEM_PROMPT", UFC_HUNT_SYSTEM_PROMPT),
            verbose=verbose,
        )
        return replace(
            configured,
            cwd=configured.cwd.resolve(),
            repository_root=configured.repository_root.resolve(),
            mcp_cwd=configured.mcp_cwd.resolve(),
        )
