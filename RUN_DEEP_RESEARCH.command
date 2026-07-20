#!/bin/zsh

set -euo pipefail

PROJECT_DIR="${0:A:h}"
SEARXNG_COMPOSE="$PROJECT_DIR/searxng/docker-compose.yml"

cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  print "Error: uv command not found. Install uv: https://docs.astral.sh/uv/"
  read "?Press Enter to close this window..."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  print "Error: Docker command not found. Install Docker Desktop."
  read "?Press Enter to close this window..."
  exit 1
fi

if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  print "Created config.yaml"
fi

# True when the Docker daemon answers within 5 seconds. A wedged daemon accepts the
# connection but never replies, so the check itself must be time-capped.
docker_responds() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 5 docker info >/dev/null 2>&1
  else
    docker info >/dev/null 2>&1 &
    local pid=$!
    for _ in {1..5}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
      return 1
    fi
    wait "$pid"
  fi
}

print "Preparing environment..."
uv sync --all-groups

print "Starting local SearXNG..."
if curl --silent --fail --max-time 5 "http://localhost:8080/search?q=test&format=json" >/dev/null 2>&1; then
  print "SearXNG is already running; skipping Docker startup."
else
  if ! docker_responds; then
    if pgrep -q com.docker.backend 2>/dev/null; then
      # Docker Desktop is running but the daemon is wedged — restart it.
      print "Docker daemon is not responding; restarting Docker Desktop..."
      killall com.docker.backend 2>/dev/null || true
      sleep 5
      { pgrep -q com.docker.backend 2>/dev/null && killall -9 com.docker.backend 2>/dev/null; } || true
      sleep 2
    else
      print "Docker daemon is not running; starting Docker Desktop..."
    fi
    open -a Docker
    DOCKER_UP=0
    for _ in {1..24}; do
      sleep 5
      if docker_responds; then
        DOCKER_UP=1
        break
      fi
    done
    if [[ "$DOCKER_UP" -ne 1 ]]; then
      print "Error: Docker daemon did not become ready within 120 seconds."
      print "Open Docker Desktop manually and run this script again."
      read "?Press Enter to close this window..."
      exit 1
    fi
    print "Docker daemon is ready."
  fi

  # A wedged Docker daemon makes `docker compose` hang forever — cap the wait.
  COMPOSE_STATUS=0
  if command -v timeout >/dev/null 2>&1; then
    timeout 90 docker compose -f "$SEARXNG_COMPOSE" up -d || COMPOSE_STATUS=$?
  else
    docker compose -f "$SEARXNG_COMPOSE" up -d &
    COMPOSE_PID=$!
    for _ in {1..90}; do
      kill -0 "$COMPOSE_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$COMPOSE_PID" 2>/dev/null; then
      kill "$COMPOSE_PID" 2>/dev/null
      COMPOSE_STATUS=124
    else
      wait "$COMPOSE_PID" || COMPOSE_STATUS=$?
    fi
  fi
  if [[ "$COMPOSE_STATUS" -ne 0 ]]; then
    print "Error: Docker did not start SearXNG (docker compose failed or timed out after 90s)."
    print "Restart Docker Desktop (or run: killall 'Docker Desktop'; open -a Docker) and try again."
    read "?Press Enter to close this window..."
    exit 1
  fi
fi

for attempt in {1..20}; do
  if curl --silent --fail "http://localhost:8080/search?q=test&format=json" >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" -eq 20 ]]; then
    print "Error: SearXNG did not start. Check Docker Desktop."
    read "?Press Enter to close this window..."
    exit 1
  fi
  sleep 1
done

print "Checking Ollama, structured inference, search, and storage..."
if ! uv run deep-research --config config.yaml smoke; then
  print "Error: environment smoke check failed. Research was not started."
  read "?Press Enter to close this window..."
  exit 1
fi

RESUME_ID="$(uv run deep-research resumable 2>/dev/null | tail -n 1 | tr -d '[:space:]' || true)"
if [[ -n "$RESUME_ID" ]]; then
  print ""
  print "Found an interrupted run that can be continued: $RESUME_ID"
  read "?Resume it instead of starting a new topic? [y/N]: " ANSWER
  if [[ "${ANSWER:l}" == y || "${ANSWER:l}" == yes ]]; then
    print ""
    print "Resuming $RESUME_ID..."
    uv run deep-research resume "$RESUME_ID"
    print ""
    print "Done. Raw data saved to:"
    print "$PROJECT_DIR/data/research.sqlite3"
    read "?Press Enter to close this window..."
    exit 0
  fi
fi

print ""
read "?Research topic: " QUERY

if [[ -z "${QUERY// }" ]]; then
  print "No topic entered. Run cancelled."
  read "?Press Enter to close this window..."
  exit 1
fi

print ""
print "Starting research..."
uv run deep-research run "$QUERY"

print ""
print "Done. Raw data saved to:"
print "$PROJECT_DIR/data/research.sqlite3"
read "?Press Enter to close this window..."
