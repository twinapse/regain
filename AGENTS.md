# Coding agent instructions

This file contains instructions for coding agents working on this repository.

Shared contributor policies such as formatting, typing, docstrings, imports, naming, and Git conventions live in
[CONTRIBUTING.md](CONTRIBUTING.md). Follow that document for all rules that apply to both humans and agents.

Reference documentation, including project context and architecture guidance, lives in the [docs](docs) directory.
[CONTRIBUTING.md](CONTRIBUTING.md#documentation) briefly explains each file in that directory.

Review the relevant reference documentation before changing related behavior.

## Python environment setup

Before running any Python-related command, activate the repository virtual environment specified by the `VIRTUAL_ENV` 
environment variable. If `VIRTUAL_ENV` is not available, or if activation fails for any reason, stop immediately and 
exit without proceeding further. Only continue with the task after activation succeeds, and show a message confirming 
that activation succeeded.

Do not install Python packages globally. Install or run tools only inside the activated virtual environment.

## Agent workflow

- Keep diffs minimal.
- Do not reformat unrelated code.
- Limit changes to the files and behavior required for the task.
- Treat these instructions as additive to the shared repository policies in [CONTRIBUTING.md](CONTRIBUTING.md).
