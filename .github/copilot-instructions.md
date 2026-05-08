# GitHub Copilot instructions

## Python environment setup

Before running any Python-related command, activate the repository virtual environment.

Resolve the virtual environment path in this order:

1. Use the path specified in `.codex/config.toml`, if present.
2. Otherwise, use the path specified in `.claude/settings.local.json`, if present.

If neither file contains a virtual environment path, stop and report that the virtual environment path could not be 
found. Do not guess, do not assume `.venv`, and do not run Python-related commands without an activated environment.

If the resolved path is relative, treat it as relative to the repository root.

Activate the environment in the same shell session before running commands:

```bash
export VIRTUAL_ENV="<resolved-virtual-env-path>"
source "$VIRTUAL_ENV/bin/activate"
```

Do not install Python packages globally. Install or run tools only inside the activated virtual environment.
