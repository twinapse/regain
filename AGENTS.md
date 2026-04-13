# Instructions for AI agents

This file contains instructions for AI agents working on this repository.

Shared contributor policies such as formatting, typing, docstrings, imports, naming, and Git conventions live in
[CONTRIBUTING.md](CONTRIBUTING.md). Follow that document for all rules that apply to both humans and agents.

Reference documentation, including project context and architecture guidance, lives in the [docs](docs) directory.
[CONTRIBUTING.md](CONTRIBUTING.md#documentation) briefly explains each file in that directory.

Review the relevant reference documentation before changing related behavior.

## Environment setup

1. Read the `VIRTUAL_ENV` environment variable and activate the Python environment it specifies.
2. If `VIRTUAL_ENV` is not available, or if activation fails for any reason, stop immediately and exit without
   proceeding further.
3. Only continue with the task after activation succeeds, and show a message confirming that activation succeeded.

## Agent workflow

- Keep diffs minimal.
- Do not reformat unrelated code.
- Limit changes to the files and behavior required for the task.
- Treat these instructions as additive to the shared repository policies in [CONTRIBUTING.md](CONTRIBUTING.md).
