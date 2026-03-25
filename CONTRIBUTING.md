# Contributing

Thank you for contributing to LLMFlow Search.

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused and describe the reason for the change in the pull request.
3. Follow the existing Python project structure and keep runtime configuration in `config.json` or environment variables rather than hardcoding values.
4. Run the available test suite before opening a pull request.

```bash
python -m pytest tests -q
```

Use clear commit messages and include documentation updates when behavior or commands change. Bug reports and feature requests can use the templates in `.github/ISSUE_TEMPLATE/`.