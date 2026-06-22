#!/usr/bin/env python3
"""Patch MiniMax M3 chat template with a narrow adaptive trivial-turn gate."""

from __future__ import annotations

import os
from pathlib import Path

MARKER = "adaptive_trivial_gate"
MODEL_DIR = "models--cyankiwi--MiniMax-M3-AWQ-INT4"

GATE = r'''{#- adaptive_trivial_gate: obvious trivial turns should skip hidden thinking. -#}
{%- if thinking_mode is defined -%}
    {%- set requested_thinking_mode = thinking_mode -%}
{%- else -%}
    {%- set requested_thinking_mode = "adaptive" -%}
{%- endif -%}
{%- set adaptive_gate = namespace(disable=false) -%}
{%- if requested_thinking_mode == "adaptive" and conversation_messages -%}
    {%- set last_message = conversation_messages | last -%}
    {%- if last_message.role == 'user' -%}
        {%- set last_text = visible_text(last_message.content) | trim | lower -%}
        {%- set compact_text = last_text | replace(' ', '') -%}
        {%- if last_text in ["hi", "hi!", "hello", "hello!", "hey", "hey!", "thanks", "thank you", "ok", "okay"] or compact_text in ["2+2", "what's2+2?", "whatis2+2?"] -%}
            {%- set adaptive_gate.disable = true -%}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}
{%- if adaptive_gate.disable -%}
    {%- set effective_thinking_mode = "disabled" -%}
{%- else -%}
    {%- set effective_thinking_mode = requested_thinking_mode -%}
{%- endif -%}
'''

OLD_MODE_BLOCK = r'''    {%- if thinking_mode is defined -%}
        {%- if thinking_mode == "enabled" -%}
            {{- 'Current thinking mode: enabled. You MUST think step by step before every response, including after receiving function/tool results.\n' }}
        {%- elif thinking_mode == "disabled" -%}
            {{- 'Current thinking mode: disabled. Do not output any thinking process.\n' }}
        {%- elif thinking_mode == "adaptive" -%}
            {{- 'Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n' }}
        {%- endif -%}
    {%- else -%}
        {{- 'Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n' }}
    {%- endif -%}
'''

NEW_MODE_BLOCK = r'''    {%- if effective_thinking_mode == "enabled" -%}
        {{- 'Current thinking mode: enabled. You MUST think step by step before every response, including after receiving function/tool results.\n' }}
    {%- elif effective_thinking_mode == "disabled" -%}
        {{- 'Current thinking mode: disabled. Do not output any thinking process.\n' }}
    {%- else -%}
        {{- 'Current thinking mode: adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing function/tool results.\n' }}
    {%- endif -%}
'''

OLD_PROMPT_BLOCK = r'''{%- if thinking_mode is defined and thinking_mode == "disabled" -%}
    {{- think_end_token }}
{%- elif thinking_mode is defined and thinking_mode == "adaptive" -%}
    {#- adaptive: no prefix, let model decide -#}
{%- elif thinking_mode is defined and thinking_mode == "enabled" -%}
    {#- enabled or not defined: default to think -#}
    {{- think_begin_token }}
{%- else -%}
    {#- adaptive: no prefix, let model decide -#}
{%- endif -%}
'''

NEW_PROMPT_BLOCK = r'''{%- if effective_thinking_mode == "disabled" -%}
    {{- think_end_token }}
{%- elif effective_thinking_mode == "enabled" -%}
    {{- think_begin_token }}
{%- else -%}
    {#- adaptive: no prefix, let model decide -#}
{%- endif -%}
'''


def cache_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (os.environ.get("HF_HOME"), "/root/.cache/huggingface", str(Path.home() / ".cache/huggingface")):
        if raw:
            path = Path(raw).expanduser()
            if path not in roots:
                roots.append(path)
    return roots


def patch_template(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"[fix-minimax-m3-reasoning-45718] chat template already patched: {path}")
        return False

    anchor = '''{%- endif -%}\n{#- Render system sp (higher priority, root role only) -#}\n'''
    if anchor not in text:
        raise RuntimeError(f"template anchor not found in {path}")
    if OLD_MODE_BLOCK not in text:
        raise RuntimeError(f"thinking mode block not found in {path}")
    if OLD_PROMPT_BLOCK not in text:
        raise RuntimeError(f"generation prompt block not found in {path}")

    text = text.replace(anchor, "{%- endif -%}\n" + GATE + "{#- Render system sp (higher priority, root role only) -#}\n", 1)
    text = text.replace(OLD_MODE_BLOCK, NEW_MODE_BLOCK, 1)
    text = text.replace(OLD_PROMPT_BLOCK, NEW_PROMPT_BLOCK, 1)
    path.write_text(text)
    print(f"[fix-minimax-m3-reasoning-45718] patched chat template: {path}")
    return True


def main() -> None:
    templates: list[Path] = []
    for root in cache_roots():
        snapshot_dir = root / "hub" / MODEL_DIR / "snapshots"
        try:
            templates.extend(snapshot_dir.glob("*/chat_template.jinja"))
        except PermissionError:
            print(f"[fix-minimax-m3-reasoning-45718] cannot read {snapshot_dir}; skipping")
    if not templates:
        print("[fix-minimax-m3-reasoning-45718] no MiniMax M3 chat templates found; skipping")
        return
    for template in templates:
        patch_template(template)


if __name__ == "__main__":
    main()
