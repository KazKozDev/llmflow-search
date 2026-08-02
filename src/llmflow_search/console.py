"""Muted ANSI terminal styling for the agent's structured progress log."""

from __future__ import annotations

import builtins
import os
import re
import sys
from typing import TextIO

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

INK = (190, 196, 207)
MUTED = (120, 129, 145)
BLUE = (112, 143, 181)
CYAN = (105, 157, 164)
VIOLET = (145, 132, 174)
AMBER = (178, 153, 102)
GREEN = (113, 157, 130)
RED = (184, 112, 121)

_STAGE_COLORS = {
    "REQUIREMENTS": VIOLET,
    "PLAN": BLUE,
    "EXEC": CYAN,
    "DRILLDOWN": CYAN,
    "POST-BATCH": BLUE,
    "LEDGER": AMBER,
    "CHALLENGE": VIOLET,
    "REEXTRACT": AMBER,
    "EVAL": BLUE,
    "REFLECT": VIOLET,
    "REPLAN": BLUE,
    "ASSIMILATE": GREEN,
    "STRATEGY": VIOLET,
    "ANSWER": CYAN,
    "VERIFY": GREEN,
}
_STAGE_RE = re.compile(r"\[([A-Z][A-Z-]+)\]")
_TOOL_RE = re.compile(r"^(\s*)\[([a-z][a-z0-9_-]+)\](.*)$", re.DOTALL)
_MODEL_ROW_RE = re.compile(r"^(\s*)(\d+\.)(\s+)(.*?)(\s+\d+(?:\.\d+)? GB)\s*$")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def color_enabled(stream: TextIO | None = None) -> bool:
    """Honor NO_COLOR and only emit ANSI to a terminal unless explicitly forced."""
    if _truthy_env("LLMFLOW_SEARCH_FORCE_COLOR"):
        return True
    if "NO_COLOR" in os.environ:
        return False
    stream = stream or sys.stdout
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text: str, color: tuple[int, int, int], *, bold: bool = False, dim: bool = False) -> str:
    if not text:
        return text
    prefix = f"\033[38;2;{color[0]};{color[1]};{color[2]}m"
    if bold:
        prefix += BOLD
    if dim:
        prefix += DIM
    return f"{prefix}{text}{RESET}"


def _style_stage_line(text: str) -> str | None:
    match = _STAGE_RE.search(text)
    if not match or match.group(1) not in _STAGE_COLORS:
        return None
    stage = match.group(1)
    before = text[: match.start()]
    label = _paint(match.group(0), _STAGE_COLORS[stage], bold=True)
    after = text[match.end() :]
    return before + label + after


def _style_text(text: str) -> str:
    staged = _style_stage_line(text)
    if staged is not None:
        return staged

    stripped = text.lstrip("\n")
    leading_newlines = text[: len(text) - len(stripped)]
    if stripped.startswith("[!]"):
        return leading_newlines + _paint("[!]", RED, bold=True) + stripped[3:]

    tool = _TOOL_RE.match(text)
    if tool:
        return tool.group(1) + _paint(f"[{tool.group(2)}]", CYAN, bold=True) + tool.group(3)

    model_row = _MODEL_ROW_RE.match(text)
    if model_row:
        return (
            model_row.group(1)
            + _paint(model_row.group(2), BLUE, bold=True)
            + model_row.group(3)
            + model_row.group(4)
            + model_row.group(5)
        )

    if "✓" in text:
        return text.replace("✓", _paint("✓", GREEN, bold=True), 1)
    if "✗" in text:
        return text.replace("✗", _paint("✗", RED, bold=True), 1)
    if text.lstrip().startswith("→"):
        position = text.index("→")
        return text[:position] + _paint("→", GREEN, bold=True) + text[position + 1 :]

    heading_prefixes = (
        "Ollama models",
        "Using:",
        "Fast model:",
        "Connecting to MCP server",
        "Sources:",
        "Coverage note",
        "Debug report:",
        "PDF report:",
    )
    if stripped.startswith(heading_prefixes):
        label, separator, rest = stripped.partition(":")
        if separator:
            return leading_newlines + _paint(label + separator, BLUE, bold=True) + rest
        return leading_newlines + _paint(stripped, BLUE, bold=True)

    compact = stripped.strip()
    if compact and set(compact) <= {"=", "─", "-"}:
        return leading_newlines + _paint(stripped, MUTED, dim=True)
    if stripped in {"Bye!", "Interactive mode. Type 'exit' to quit."}:
        return text
    return text


def print(  # noqa: A001 - intentionally mirrors builtins.print for drop-in use
    *objects,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Drop-in print replacement that styles known progress-log shapes."""
    stream = file or sys.stdout
    text = sep.join(str(item) for item in objects)
    if color_enabled(stream):
        text = _style_text(text)
    builtins.print(text, end=end, file=stream, flush=flush)
