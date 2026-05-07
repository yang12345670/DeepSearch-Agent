"""ReAct Executor — LLM-driven tool-use loop.

Flow:
    user query
      └► LLM (with tools=[search_knowledge_base, generate_pdf, ...])
           ├─► tool_calls ─► dispatch ─► tool_response ─► back to LLM
           └─► final content ─► return

Round budget is bounded by `max_rounds` (default 5). Each tool error is
captured into the tool_response so the LLM can decide to retry/abort.

Artifacts (e.g. generated PDF file paths) are collected separately so the
API layer can surface them to the frontend (download links etc.) without
parsing the LLM text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.tools import get_openai_tools_schema, get_tool, load_builtin_tools
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """\
你是一个金融研究 agent，可以调用工具帮助回答用户问题。

可用工具：
- search_knowledge_base(query): 检索金融知识库（SEC 财报 + 财经新闻），返回带引用的答案
- generate_pdf(title, sections): 生成结构化 PDF 报告

工作准则：
1. 用户问需要事实信息时，先用 search_knowledge_base 获取证据再回答；不要凭空回答
2. 多角度问题分多次 search_knowledge_base，每次只问一个明确角度（如风险、业绩、展望各一次）
3. 用户明确要 PDF / 报告 / 导出文件时：先 search 收集所需信息，再调 generate_pdf 把内容组织成 sections=[{heading, body, citations}]
4. 工具结果中的 citations 字段是关键依据，最终回答要点上来源
5. 简洁、有据，不要重复 search 同一个角度
6. 如果 search 多次都拿不到相关证据，直接告诉用户"没有相关资料"，不要编造
"""


@dataclass
class ToolCallTrace:
    name: str
    arguments: Dict[str, Any]
    response: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ReActResult:
    answer: str
    rounds: int
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    debug_trace: str = ""


class ReActExecutor:
    """LLM-driven ReAct loop with pluggable tool registry."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_rounds: int = 5,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        load_builtin_tools()  # idempotent — populates TOOLS_REGISTRY
        self._tools_schema = get_openai_tools_schema()
        logger.info(
            "ReActExecutor ready: max_rounds=%d, tools=%s",
            max_rounds, [t["function"]["name"] for t in self._tools_schema],
        )

    def run(self, user_query: str) -> ReActResult:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        tool_traces: List[ToolCallTrace] = []
        artifacts: List[Dict[str, Any]] = []
        debug_lines: List[str] = []

        for round_idx in range(self.max_rounds):
            logger.info("ReAct round %d/%d", round_idx + 1, self.max_rounds)
            try:
                assistant_msg = self.llm.generate_with_tools(
                    messages=messages,
                    tools=self._tools_schema,
                )
            except NotImplementedError as e:
                logger.error("ReAct: provider does not support tool calling: %s", e)
                return ReActResult(
                    answer=f"[ReAct 不可用] {e}",
                    rounds=round_idx,
                    debug_trace="provider does not support tool calling",
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("ReAct: LLM call failed")
                return ReActResult(
                    answer=f"[LLM 调用失败] {e}",
                    rounds=round_idx,
                    debug_trace=f"llm error: {e}",
                )

            # Append assistant message (drop None fields so the SDK is happy)
            messages.append({k: v for k, v in assistant_msg.items() if v is not None})

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                # Final answer reached
                final_content = assistant_msg.get("content") or ""
                debug_lines.append(f"Round {round_idx + 1}: final answer ({len(final_content)} chars)")
                return ReActResult(
                    answer=final_content,
                    rounds=round_idx + 1,
                    tool_calls=tool_traces,
                    artifacts=artifacts,
                    debug_trace="\n".join(debug_lines),
                )

            # Execute each tool call, append tool response to messages
            debug_lines.append(
                f"Round {round_idx + 1}: {len(tool_calls)} tool call(s)"
            )
            for tc in tool_calls:
                tc_id = tc["id"]
                fn_name = tc["function"]["name"]
                arg_str = tc["function"]["arguments"] or "{}"

                # Parse arguments JSON (LLM occasionally emits malformed JSON)
                try:
                    args = json.loads(arg_str)
                    parse_error: Optional[str] = None
                except json.JSONDecodeError as e:
                    args = {}
                    parse_error = f"argument JSON parse error: {e}"

                # Dispatch
                if parse_error:
                    response = {"error": parse_error}
                else:
                    tool = get_tool(fn_name)
                    if tool is None:
                        response = {"error": f"Unknown tool: {fn_name}"}
                    else:
                        response = tool.execute(**args)

                tool_traces.append(ToolCallTrace(
                    name=fn_name,
                    arguments=args,
                    response=response,
                    error=parse_error or response.get("error"),
                ))

                # Side effect: collect artifacts for the API layer
                if fn_name == "generate_pdf" and "file_path" in response:
                    artifacts.append({
                        "type": "pdf",
                        "file_path": response["file_path"],
                        "size_bytes": response.get("size_bytes"),
                    })

                debug_lines.append(
                    f"  - {fn_name}({_brief_args(args)}) "
                    f"→ {'ERR' if 'error' in response else 'OK'}"
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": fn_name,
                    "content": json.dumps(response, ensure_ascii=False),
                })

        # Hit the round budget — return what we have
        logger.warning("ReAct hit max_rounds=%d, truncating", self.max_rounds)
        last_content = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                last_content = m["content"]
                break
        return ReActResult(
            answer=last_content or "（达到最大工具调用轮数，未能给出最终回答）",
            rounds=self.max_rounds,
            tool_calls=tool_traces,
            artifacts=artifacts,
            truncated=True,
            debug_trace="\n".join(debug_lines) + "\n[TRUNCATED at max_rounds]",
        )


def _brief_args(args: Dict[str, Any], max_len: int = 60) -> str:
    """Compact arg preview for logs."""
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."
