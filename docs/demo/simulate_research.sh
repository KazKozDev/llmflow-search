#!/bin/zsh
# Part 3: prints simulated research output after the user types a question.

set -euo pipefail

bold=$'\033[1m'
reset=$'\033[0m'
cyan=$'\033[38;2;105;157;164m'
violet=$'\033[38;2;145;132;174m'
green=$'\033[38;2;113;157;130m'
amber=$'\033[38;2;178;153;102m'
muted=$'\033[38;2;120;129;145m'
blue_head=$'\033[38;2;112;143;181m'
red=$'\033[38;2;184;112;121m'

# Separator
print -r -- ""
print -r -- "${muted}──────────────────────────────────────────────────${reset}"

# Requirements phase
sleep 0.3
print -r -- "${violet}${bold}[REQUIREMENTS]${reset} Decomposing question into verifiable conditions"
sleep 0.4
print -r -- "  ${green}${bold}→${reset} 3 conditions extracted"

# Plan phase
sleep 0.3
print -r -- "${blue_head}${bold}[PLAN]${reset} Building research plan from conditions"
sleep 0.5
print -r -- "  Step 1: Search for latest stable Python release on python.org"
print -r -- "  Step 2: Verify release date and version number"
print -r -- "  Step 3: Check release notes for key changes"

# Execution phase
sleep 0.4
print -r -- "${cyan}${bold}[EXEC]${reset} Step 1/3"
sleep 0.2
print -r -- "  ${cyan}${bold}[web_search]${reset} \"latest stable Python release 2025\""
sleep 0.6
print -r -- "  ${green}${bold}✓${reset} 8 results"
sleep 0.2
print -r -- "  ${cyan}${bold}[fetch_page]${reset} python.org/downloads/"
sleep 0.5
print -r -- "  ${green}${bold}✓${reset} 12,340 chars"

sleep 0.3
print -r -- "${cyan}${bold}[EXEC]${reset} Step 2/3"
sleep 0.2
print -r -- "  ${cyan}${bold}[fetch_page]${reset} docs.python.org/3/whatsnew/3.13.html"
sleep 0.5
print -r -- "  ${green}${bold}✓${reset} 45,120 chars"

sleep 0.3
print -r -- "${cyan}${bold}[EXEC]${reset} Step 3/3"
sleep 0.2
print -r -- "  ${cyan}${bold}[web_search]${reset} \"Python 3.13 release notes changes\""
sleep 0.5
print -r -- "  ${green}${bold}✓${reset} 6 results"

# Evidence ledger
sleep 0.3
print -r -- "${amber}${bold}[LEDGER]${reset} Evaluating evidence against conditions"
sleep 0.4
print -r -- "  Condition 1: ${green}${bold}✓${reset} supported (3 sources)"
print -r -- "  Condition 2: ${green}${bold}✓${reset} supported (2 sources)"
print -r -- "  Condition 3: ${green}${bold}✓${reset} supported (2 sources)"

# Challenge
sleep 0.3
print -r -- "${violet}${bold}[CHALLENGE]${reset} Cross-checking claims against sources"
sleep 0.4

# Answer
sleep 0.2
print -r -- "${cyan}${bold}[ANSWER]${reset} Drafting cited answer"
sleep 0.3

# Verify
print -r -- "${green}${bold}[VERIFY]${reset} All 3 conditions satisfied"
sleep 0.3

# Final answer
print -r -- ""
print -r -- "The latest stable Python release is ${bold}Python 3.13.2${reset}, released on"
print -r -- "February 4, 2025 [1]. Key changes include:"
print -r -- ""
print -r -- "  - A new interactive interpreter (REPL) with multi-line editing"
print -r -- "    and color output [2]"
print -r -- "  - An experimental JIT compiler for improved performance [2]"
print -r -- "  - Improved error messages with more helpful suggestions [3]"
print -r -- "  - Initial support for free-threaded execution (no GIL) [2]"
print -r -- ""

# Sources
print -r -- "${blue_head}${bold}Sources:${reset}"
print -r -- "  [1] Python Release Python 3.13.2 — python.org/downloads/release/python-3132/"
print -r -- "  [2] What's New in Python 3.13 — docs.python.org/3/whatsnew/3.13.html"
print -r -- "  [3] Python 3.13 Release Schedule — peps.python.org/pep-0719/"

# Reports
sleep 0.2
print -r -- "${blue_head}${bold}PDF report:${reset} reports/python_latest_release.pdf"

# Bottom separator
print -r -- "${muted}──────────────────────────────────────────────────${reset}"
print -r -- "  Steps: 5 | Type next question or 'exit'"
print -r -- "${muted}──────────────────────────────────────────────────${reset}"
