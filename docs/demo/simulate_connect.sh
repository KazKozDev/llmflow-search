#!/bin/zsh
# Part 2: prints the "after model selection" output.
# Called after the user types "2" in the VHS tape.

set -euo pipefail

bold=$'\033[1m'
reset=$'\033[0m'
cyan=$'\033[38;2;105;157;164m'
violet=$'\033[38;2;145;132;174m'
green=$'\033[38;2;113;157;130m'
amber=$'\033[38;2;178;153;102m'
muted=$'\033[38;2;120;129;145m'
blue_head=$'\033[38;2;112;143;181m'

print -r -- ""
print -r -- "${blue_head}${bold}Using:${reset} gemma4:26b-mlx"
print -r -- "${blue_head}${bold}Fast model:${reset} qwen3:8b"
print -r -- "${blue_head}${bold}Connecting to MCP server${reset} (footnote-mcp)... ${green}${bold}✓${reset} (45 tools)"
print -r -- "  Profile: footnote"
print -r -- ""
print -r -- "${muted}==================================================${reset}"
print -r -- "  Interactive mode. Type 'exit' to quit."
print -r -- "${muted}==================================================${reset}"
print -r -- ""
