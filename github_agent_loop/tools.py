from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from copilot.tools import Tool, ToolInvocation, ToolResult
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from github_agent_loop.config import AgentConfig


class MCPToolRuntime:
    def __init__(self, config: AgentConfig):
        self._config = config

    async def _with_session(self, operation: Callable[[ClientSession], Any]) -> Any:
        params = StdioServerParameters(
            command=self._config.mcp_command,
            args=list(self._config.mcp_args),
            cwd=self._config.mcp_cwd,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await operation(session)

    async def list_tools_async(self) -> list[dict[str, Any]]:
        async def _list(session: ClientSession) -> list[dict[str, Any]]:
            result = await session.list_tools()
            payload = result.model_dump() if hasattr(result, "model_dump") else result
            return payload["tools"]

        return await self._with_session(_list)

    def list_tools(self) -> list[dict[str, Any]]:
        return asyncio.run(self.list_tools_async())

    async def call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        async def _call(session: ClientSession) -> Any:
            result = await session.call_tool(tool_name, arguments)
            payload = result.model_dump() if hasattr(result, "model_dump") else result
            if payload.get("isError"):
                return payload

            items: list[Any] = []
            for item in payload.get("content", []):
                if item.get("type") != "text":
                    items.append(item)
                    continue
                text = item.get("text", "")
                try:
                    items.append(json.loads(text))
                except json.JSONDecodeError:
                    items.append(text)

            if len(items) == 1:
                return items[0]
            return items

        return await self._with_session(_call)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return asyncio.run(self.call_tool_async(tool_name, arguments))


async def build_sdk_tools_async(config: AgentConfig) -> list[Tool]:
    runtime = MCPToolRuntime(config)
    tool_payloads = await runtime.list_tools_async()
    sdk_tools: list[Tool] = []

    for payload in tool_payloads:
        name = payload["name"]
        description = payload.get("description") or ""
        parameters = payload.get("inputSchema") or {"type": "object", "properties": {}}

        async def handler(
            invocation: ToolInvocation,
            *,
            tool_name: str = name,
            tool_runtime: MCPToolRuntime = runtime,
        ) -> ToolResult:
            params = invocation.arguments or {}
            if hasattr(params, "model_dump"):
                params = params.model_dump()
            result = await tool_runtime.call_tool_async(tool_name, dict(params))
            return ToolResult(
                text_result_for_llm=json.dumps(result, indent=2),
                result_type="success",
            )

        sdk_tools.append(
            Tool(
                name=name,
                description=description,
                handler=handler,
                parameters=parameters,
                skip_permission=True,
            )
        )

    return sdk_tools


def build_sdk_tools(config: AgentConfig) -> list[Tool]:
    return asyncio.run(build_sdk_tools_async(config))
