# AGENTS

This file contains the playbook for AI agents working on the codebase.

## General requirements

- Treat these guidelines as the default contract for new code and refactors.
- Prefer minimal diffs: don’t reformat unrelated code.

## Code style guidelines

- **Formatting**:
  - Follow PEP 8 and Google Python Style.
  - Use 4-space indentation, 120-char lines.
  - Prefer `f`-strings; avoid `%` or `.format`.
  - Use single quotes for string literals (`'...'`) by default; use double quotes only when the string contains single 
    quotes.
  - Avoid escaping single quotes in string literals (prefer using double quotes inside single-quoted strings rather 
    than escaping quotes).
- **Typing & arguments**:
  - All functions and methods are type hinted (including return types).
  - Default to keyword arguments for multi-parameter calls and public APIs; single obvious parameters may stay 
    positional.
  - Use keyword-only parameters (`*`) when a call site benefits from named arguments.
  - Framework handlers keep framework-prescribed signatures.
- **Docstrings & comments**:
  - Use Google style (`Args`, `Returns`, `Raises`) in all functions.
    - Include type hints.
    - Indent descriptions after the colons.
  - Docstrings must use a “newline-after-opening-quotes” style: the opening `"""` is on its own line, then the summary 
    starts on the next line (no same-line docstring summaries).
  - Triple double quotes are for docstrings only—use `#` for comments.
  - Break long parameter lists across lines.
  - Lightweight banner comments are used to mark sections (e.g., `####################` headers).
- **Imports & exports**:
  - Don't use `from __future__ ...` imports.
  - Use absolute imports; avoid relative imports.
  - Group stdlib, third-party, then project imports.
  - Apply `isort` style:
    - Import only one symbol per line, except for `typing` imports.
    - Order imports alphabetically within groups.
  - Declare `__all__` after imports to document the public surface in modules that export symbols.
    - Update these when adding exports.
    - Add one line per exported symbol.
  - Keep naming consistent with snake_case for functions/vars, PascalCase for classes/enums, ALL_CAPS for constants.
