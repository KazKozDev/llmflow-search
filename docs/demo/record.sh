#!/usr/bin/env bash
# Re-record the README's demo GIF from a real research session.
#
# What replaced what, and why it matters: the previous pipeline drove three
# simulate_*.sh scripts that printed a scripted transcript, so the GIF on the front page
# was a reproduction of a run rather than a run. Every line the current GIF shows was
# produced by the agent answering the question below against the live web.
#
# Needs: asciinema and agg (brew install asciinema agg), Ollama with the model below
# pulled, and the project installed (uv sync).
#
# The recording is a real session, so it is never identical twice: pages move, search
# results reorder, the model words its answer differently. That is the point. What the
# flags fix is only the presentation — terminal size, how far idle time is compressed,
# and playback speed.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

QUESTION="${1:-what did the Python 3.13 release add for free-threading}"
MODEL="${LLMFLOW_SEARCH_MODEL:-deepseek-v4.1-flash:cloud}"
CAST="docs/demo/session.cast"
GIF="assets/llmflow-search-demo.gif"

# Keys live in .env and must not reach the recording. Nothing below prints the
# environment, and the finished cast is grepped for the key before it is rendered.
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export LLMFLOW_SEARCH_MODEL="$MODEL"
export PATH="$PWD/.venv/bin:$PATH"

echo "Recording a live session with $MODEL..."
asciinema rec --window-size 100x30 --overwrite \
  -c "docs/demo/type_question.exp '$QUESTION'" "$CAST"

# A secret in a GIF cannot be taken back once it is committed, so this check is not
# optional and not advisory: it stops the pipeline.
for var in TAVILY_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY; do
  value="${!var:-}"
  if [[ -n "$value" ]] && grep -qF -- "$value" "$CAST"; then
    echo "REFUSING TO RENDER: $var appears in $CAST" >&2
    exit 1
  fi
done

# --idle-time-limit collapses the model's thinking pauses and the search rate limiter's
# waits, which are most of a real run's wall clock and none of its interest. --speed is
# then the only compression left, kept low enough that a reader can follow the output.
agg --idle-time-limit 1.2 --speed 1.4 --font-size 16 --theme asciinema --fps-cap 12 \
  "$CAST" "$GIF"

echo "Wrote $GIF"
ls -lh "$GIF"
