# Launch the LLMFlow-Search research agent on Windows. Syncs its locked uv environment,
# installs the sibling footnote-mcp server package, ensures Chromium, then runs the agent.
$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

Write-Host ""
Write-Host "LLMFlow-Search" -ForegroundColor Blue
Write-Host "LangGraph web research agent for macOS with Ollama" -ForegroundColor DarkGray
Write-Host ""

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

$FootnoteMcpDir = Join-Path (Split-Path -Parent $ProjectDir) "footnote-mcp"
if (-not (Test-Path $FootnoteMcpDir)) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "[!] git is required to fetch footnote-mcp. Install git, or clone" -ForegroundColor Yellow
        Write-Host "    https://github.com/KazKozDev/footnote-mcp.git into $(Split-Path -Parent $ProjectDir) yourself." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "footnote-mcp not found next to this checkout - cloning it..."
    git clone --quiet https://github.com/KazKozDev/footnote-mcp.git $FootnoteMcpDir
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Ollama was not found on PATH. Install it from https://ollama.com and" -ForegroundColor Yellow
    Write-Host "    pull at least one model (e.g. ``ollama pull qwen2.5:7b``) before continuing." -ForegroundColor Yellow
}

uv python install --quiet
uv sync --locked --quiet
uv pip install --quiet --python .venv\Scripts\python.exe -e ..\footnote-mcp "mcp<2"
try { uv run --no-sync python -m playwright install chromium *> $null } catch {}

# Local secrets (API keys) live in .env, which is gitignored. Keeping them here
# rather than in the shell profile keeps them out of shell history too.
$EnvFile = Join-Path $ProjectDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=][^=]*)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
        }
    }
}

if (-not $env:LLMFLOW_SEARCH_FORCE_COLOR) {
    $env:LLMFLOW_SEARCH_FORCE_COLOR = "1"
}

uv run --no-sync python -m llmflow_search

Write-Host ""
Read-Host "Press Enter to close..." | Out-Null
