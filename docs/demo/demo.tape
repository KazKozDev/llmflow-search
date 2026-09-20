# LLMFlow-Search Demo GIF
# Generates: assets/llmflow-search-demo.gif
# Run:       vhs docs/demo/demo.tape

Output assets/llmflow-search-demo.gif

Require zsh

Set Shell "zsh"
Set FontSize 16
Set Width 920
Set Height 620
Set Padding 20
Set Framerate 30
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"
Set CursorBlink false

# ── Phase 1: Show the logo ──
Hide
Type "zsh docs/demo/simulate_demo.sh"
Enter
Show
Sleep 3s

# ── Phase 2: Pick the model ──
Sleep 0.5
Type@100ms "2"
Enter
Sleep 0.8

# ── Phase 3: MCP connection ──
Hide
Type "zsh docs/demo/simulate_connect.sh"
Enter
Show
Sleep 2.5

# ── Phase 4: Ask a question ──
Type ">>> "
Sleep 0.3
Type@60ms "What is the latest stable Python release?"
Sleep 0.5
Enter
Sleep 0.5

# ── Phase 5: Research output ──
Hide
Type "zsh docs/demo/simulate_research.sh"
Enter
Show
Sleep 12s
