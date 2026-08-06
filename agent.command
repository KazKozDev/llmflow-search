#!/bin/zsh
# Launch the LLMFlow-Search research agent. Syncs its locked uv environment, installs
# the sibling footnote-mcp server package, ensures Chromium, then runs the agent.
set -euo pipefail

PROJECT_DIR="${0:A:h}"

cd "$PROJECT_DIR"

print_logo() {
  local blue=$'\033[38;2;103;148;254m'
  local flow_blue=$'\033[38;2;81;115;205m'
  local search_gray=$'\033[38;2;174;181;193m'
  local reset=$'\033[0m'

  print -r -- "

${blue}
██╗     ██╗     ███╗   ███╗ ${flow_blue}███████╗██╗      ██████╗ ██╗    ██╗
██║     ██║     ████╗ ████║ ${flow_blue}██╔════╝██║     ██╔═══██╗██║    ██║
██║     ██║     ██╔████╔██║ ${flow_blue}█████╗  ██║     ██║   ██║██║ █╗ ██║
██║     ██║     ██║╚██╔╝██║ ${flow_blue}██╔══╝  ██║     ██║   ██║██║███╗██║
███████╗███████╗██║ ╚═╝ ██║ ${flow_blue}██║     ███████╗╚██████╔╝╚███╔███╔╝
╚══════╝╚══════╝╚═╝     ╚═╝ ${flow_blue}╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝${reset}

${search_gray}███████╗ ███████╗  █████╗  ██████╗   ██████╗ ██╗  ██╗
██╔════╝ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔════╝ ██║  ██║
███████╗ █████╗   ███████║ ██████╔╝ ██║      ███████║
╚════██║ ██╔══╝   ██╔══██║ ██╔══██╗ ██║      ██╔══██║
███████║ ███████╗ ██║  ██║ ██║  ██║ ╚██████╗ ██║  ██║
╚══════╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝ ╚═╝  ╚═╝${reset}



"
}

print_logo

if ! command -v uv >/dev/null 2>&1; then
  print -r -- "Installing uv..."
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --locked --quiet
uv pip install --quiet --python .venv/bin/python -e ../footnote-mcp "mcp<2"
uv run --no-sync python -m playwright install chromium >/dev/null 2>&1 || true

# Local secrets (API keys) live in .env, which is gitignored. Keeping them here
# rather than in the shell profile keeps them out of shell history too.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export LLMFLOW_SEARCH_FORCE_COLOR="${LLMFLOW_SEARCH_FORCE_COLOR:-1}"
uv run --no-sync python -m llmflow_search

echo ""
read -r "?Press Enter to close..."
