#!/bin/zsh
# Full demo session script — run inside asciinema rec for a realistic capture.
# Simulates: logo → model selection → MCP connect → question → research → answer

set -euo pipefail

DEMO_DIR="${0:A:h}"

# ── 1. Logo + model list ──────────────────────────────────────────────────────
zsh "$DEMO_DIR/simulate_demo.sh"

# ── 2. Simulate user typing "2" and pressing Enter ───────────────────────────
sleep 1.2
print -r -- "2"

sleep 0.3
zsh "$DEMO_DIR/simulate_connect.sh"

# ── 3. Simulate user typing the question ─────────────────────────────────────
sleep 0.8
printf ">>> "
sleep 0.3

# Type character by character to look real
question="What is the latest stable Python release?"
for ((i=1; i<=${#question}; i++)); do
  printf "%s" "${question[i]}"
  sleep 0.05
done
print -r -- ""

sleep 0.4

# ── 4. Research output ───────────────────────────────────────────────────────
zsh "$DEMO_DIR/simulate_research.sh"

sleep 3
