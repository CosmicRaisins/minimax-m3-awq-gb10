# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from typing import Any
import xml.etree.ElementTree as ET

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_tool_properties,
)

logger = init_logger(__name__)


class MinimaxM3ToolParser(ToolParser):
    """Python fallback parser for MiniMax M3's namespaced XML tool calls."""

    supports_required_and_named = False

    ns_token = "]<]minimax[>["
    tool_call_start_token = ns_token + "<tool_call>"
    tool_call_end_token = ns_token + "</tool_call>"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.tool_call_complete_regex = re.compile(
            re.escape(self.tool_call_start_token)
            + r"(.*?)"
            + re.escape(self.tool_call_end_token),
            re.DOTALL,
        )
        self.invoke_complete_regex = re.compile(
            r"<invoke\s+name=(['\"])(.*?)\1\s*>(.*?)</invoke>",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
        return request

    def _denamespace(self, text: str) -> str:
        return text.replace(self.ns_token, "")

    def _partial_tag_overlap(self, text: str, tag: str) -> int:
        max_check = min(len(tag) - 1, len(text))
        for length in range(max_check, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def _schema_for_child(self, schema: dict[str, Any], key: str) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        props = schema.get("properties")
        if isinstance(props, dict):
            child = props.get(key)
            if isinstance(child, dict):
                return child
        return {}

    def _schema_for_array_item(self, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        item_schema = schema.get("items")
        return item_schema if isinstance(item_schema, dict) else {}

    def _coerce_text(self, value: str, schema: dict[str, Any]) -> Any:
        value = value.strip()
        if not schema:
            return value
        try:
            types = extract_types_from_schema(schema)
            return coerce_to_schema_type(value, types)
        except Exception:
            logger.debug("MiniMax M3 could not coerce tool value", exc_info=True)
            return value

    def _element_to_value(self, elem: ET.Element, schema: dict[str, Any]) -> Any:
        children = list(elem)
        if not children:
            return self._coerce_text(elem.text or "", schema)

        if all(child.tag == "item" for child in children):
            item_schema = self._schema_for_array_item(schema)
            return [self._element_to_value(child, item_schema) for child in children]

        result: dict[str, Any] = {}
        for child in children:
            child_schema = self._schema_for_child(schema, child.tag)
            value = self._element_to_value(child, child_schema)
            if child.tag in result:
                current = result[child.tag]
                if not isinstance(current, list):
                    result[child.tag] = [current]
                result[child.tag].append(value)
            else:
                result[child.tag] = value
        return result

    def _parse_invokes_with_xml(self, block: str) -> list[tuple[str, dict[str, Any]]]:
        clean = self._denamespace(block)
        root = ET.fromstring(f"<root>{clean}</root>")
        parsed: list[tuple[str, dict[str, Any]]] = []

        for invoke in root.findall("invoke"):
            function_name = invoke.attrib.get("name", "").strip()
            if not function_name:
                continue

            tool_properties = find_tool_properties(self.tools, function_name)
            args: dict[str, Any] = {}
            for param in list(invoke):
                schema = tool_properties.get(param.tag, {})
                args[param.tag] = self._element_to_value(param, schema)
            parsed.append((function_name, args))

        return parsed

    def _parse_invokes_with_regex(self, block: str) -> list[tuple[str, dict[str, Any]]]:
        clean = self._denamespace(block)
        parsed: list[tuple[str, dict[str, Any]]] = []
        for _, function_name, body in self.invoke_complete_regex.findall(clean):
            tool_properties = find_tool_properties(self.tools, function_name)
            args: dict[str, Any] = {}
            for param_match in re.finditer(
                r"<([A-Za-z_][\w.\-]*)>(.*?)</\1>", body, re.DOTALL
            ):
                key = param_match.group(1)
                value = param_match.group(2)
                schema = tool_properties.get(key, {})
                args[key] = self._coerce_text(value, schema)
            parsed.append((function_name, args))
        return parsed

    def _parse_invokes(self, block: str) -> list[tuple[str, dict[str, Any]]]:
        try:
            return self._parse_invokes_with_xml(block)
        except ET.ParseError:
            logger.debug(
                "MiniMax M3 XML parse failed; falling back to regex parse",
                exc_info=True,
            )
            return self._parse_invokes_with_regex(block)

    def _tool_calls_from_blocks(self, blocks: list[str]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for block in blocks:
            for function_name, args in self._parse_invokes(block):
                tool_calls.append(
                    ToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps(args, ensure_ascii=False),
                        ),
                    )
                )
        return tool_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            blocks = self.tool_call_complete_regex.findall(model_output)
            tool_calls = self._tool_calls_from_blocks(blocks)
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            self.prev_tool_call_arr.clear()
            for tool_call in tool_calls:
                self.prev_tool_call_arr.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                )

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx].strip() or None
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )
        except Exception:
            logger.exception("Error extracting MiniMax M3 tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _complete_invoke_blocks(self, current_text: str) -> list[str]:
        if self.tool_call_start_token not in current_text:
            return []
        tail = current_text.split(self.tool_call_start_token, 1)[1]
        tail = tail.split(self.tool_call_end_token, 1)[0]
        clean = self._denamespace(tail)
        return [match.group(0) for match in self.invoke_complete_regex.finditer(clean)]

    def _extract_delta_tool_calls(self, current_text: str) -> list[DeltaToolCall]:
        complete_invokes = self._complete_invoke_blocks(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_block = complete_invokes[self.current_tool_index]
            parsed = self._parse_invokes(invoke_block)
            self.current_tool_index += 1
            if not parsed:
                continue

            function_name, args = parsed[0]
            args_json = json.dumps(args, ensure_ascii=False)
            idx = len(self.prev_tool_call_arr)
            self.prev_tool_call_arr.append(
                {"name": function_name, "arguments": args_json}
            )
            self.streamed_args_for_tool.append(args_json)
            delta_tool_calls.append(
                DeltaToolCall(
                    index=idx,
                    id=make_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=function_name,
                        arguments=args_json,
                    ),
                )
            )

        return delta_tool_calls

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_token_ids, current_token_ids, delta_token_ids, request

        start_in_current = self.tool_call_start_token in current_text
        start_in_previous = self.tool_call_start_token in previous_text
        start_in_delta = self.tool_call_start_token in delta_text
        tool_call_starting = start_in_current and not start_in_previous

        if not previous_text or tool_call_starting:
            self.current_tool_index = 0
            self.prev_tool_call_arr.clear()
            self.streamed_args_for_tool.clear()
            self.is_tool_call_started = start_in_current

        if not self.is_tool_call_started:
            overlap = self._partial_tag_overlap(current_text, self.tool_call_start_token)
            if overlap:
                safe_delta_len = max(0, len(delta_text) - overlap)
                safe_delta = delta_text[:safe_delta_len]
                return DeltaMessage(content=safe_delta) if safe_delta else None
            return DeltaMessage(content=delta_text) if delta_text else None

        content_before = None
        if start_in_delta:
            before = delta_text[: delta_text.index(self.tool_call_start_token)]
            content_before = before or None

        delta_tool_calls = self._extract_delta_tool_calls(current_text)
        if delta_tool_calls or content_before:
            return DeltaMessage(
                content=content_before,
                tool_calls=delta_tool_calls,
            )

        if (
            self.tool_call_end_token in current_text
            and self.prev_tool_call_arr
            and not delta_text
        ):
            return DeltaMessage(content="")

        return None
