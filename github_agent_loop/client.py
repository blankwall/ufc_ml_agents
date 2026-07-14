from __future__ import annotations

import asyncio
from typing import Any

from copilot import CopilotClient, ExternalServerConfig, SubprocessConfig
from copilot.generated.session_events import SessionEventType
from copilot.session import PermissionHandler, SessionEvent

from github_agent_loop.config import AgentConfig
from github_agent_loop.tools import build_sdk_tools_async


class GitHubModelsClient:
    def __init__(self, config: AgentConfig):
        self._config = config

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return asyncio.run(self._complete(messages))

    async def _complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self._latest_user_prompt(messages)
        used_tools: set[str] = set()

        def on_event(event: SessionEvent) -> None:
            if event.type != SessionEventType.TOOL_EXECUTION_START:
                return
            data = event.data
            tool_name = getattr(data, "mcp_tool_name", None) or getattr(data, "tool_name", None)
            if tool_name:
                used_tools.add(tool_name)

        client = CopilotClient(self._copilot_transport())
        await client.start()
        session = await client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model=self._config.model,
            system_message={"mode": "append", "content": self._config.system_prompt},
            working_directory=str(self._config.cwd),
            streaming=False,
            tools=await build_sdk_tools_async(self._config),
            on_event=on_event,
        )
        try:
            final_content = await self._send_turn(session, prompt)
            for _ in range(self._config.max_reminders):
                missing = [tool for tool in self._config.required_tools if tool not in used_tools]
                if not missing:
                    break
                reminder = (
                    "Continue the UFC MCP hunt before finalizing. "
                    f"Still missing these evidence lanes: {', '.join(missing)}. "
                    "Use the remaining MCP tools now, then finish the analysis."
                )
                final_content = await self._send_turn(session, reminder)

            return {
                "role": "assistant",
                "content": final_content,
                "used_tools": sorted(used_tools),
            }
        finally:
            await session.disconnect()
            await client.stop()

    async def _send_turn(self, session, prompt: str) -> str:
        event = await session.send_and_wait(prompt, timeout=self._config.request_timeout_seconds)
        if event is None:
            return ""
        return getattr(event.data, "content", "") or ""

    def _copilot_transport(self) -> SubprocessConfig | ExternalServerConfig:
        if self._config.cli_url:
            return ExternalServerConfig(url=self._config.cli_url)

        return SubprocessConfig(
            cwd=str(self._config.cwd),
            cli_path=self._config.cli_path,
            use_stdio=True,
            use_logged_in_user=True,
        )

    @staticmethod
    def _latest_user_prompt(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            role = getattr(message, "type", None) or message.get("role")
            if role == "human" or role == "user":
                return getattr(message, "content", None) or message.get("content", "")
        raise ValueError("No user prompt found in message history.")
