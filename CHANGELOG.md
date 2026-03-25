# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to semantic versioning when version tags are introduced.

## [Unreleased]

### Added

- No unreleased entries yet.

### Changed

- No unreleased entries yet.

### Fixed

- No unreleased entries yet.

### Removed

- No unreleased entries yet.

## [2.1.0] - 2026-03-25

### Added

- Project documentation set with setup guide, contribution guide, security
	policy, changelog scaffold, and GitHub issue and pull request templates.
- Focused regression tests for planning, report generation, background jobs,
	web server routes, and tool result handling.
- Local environment example file for provider configuration.

### Changed

- Refreshed README to better describe architecture, setup, API surface, and
	runtime workflows.
- Updated configuration models for Pydantic v2 style validators and cleaner
	typed field declarations.
- Refactored planning and report generation modules for clearer structure and
	improved maintainability without changing intended behavior.

### Fixed

- Restored follow-up search plan revision when `AgentCore` stores DuckDuckGo
	results using `search_duckduckgo` as the source name.
- Fixed Wikipedia URL parsing in the DuckDuckGo result pipeline by using the
	actual Wikipedia tool implementation.
- Fixed report generation so collected `search_results` snippets are included in
	the LLM context instead of producing false "no sources provided" narratives.
- Pinned `chardet<6` to remove the startup dependency warning from `requests`.

### Removed

- Temporary local WebSocket verification helper used during debugging.