Codex CLI Rules (Project-local)

- Be concise, direct, and friendly. Prefer actionable guidance to long explanations.
- Always describe tool actions briefly before running them when grouped.
- Use the plan tool for multi-step or ambiguous tasks; skip for trivial tasks.
- Keep changes minimal and focused; avoid unrelated edits or refactors.
- Prefer root-cause fixes over surface patches; match project style.
- Use `apply_patch` for all edits; don’t introduce external editors.
- Read files in chunks (<=250 lines). Prefer `rg` to search.
- In workspace-write mode, write only within the repo; avoid destructive actions.
- When running commands, validate changes thoughtfully (tests/builds) when applicable.
- Provide short progress updates during long tasks; summarize results clearly.
- Use inline code formatting for paths/commands (e.g., `src/graph.py`).
- Avoid adding licenses or headers unless requested.
- Don’t expose API keys or secrets; keep network use minimal and explicit.

