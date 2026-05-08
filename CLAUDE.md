@AGENTS.md

# Claude Code instructions

This repository uses `AGENTS.md` as the authoritative source of project instructions.

At the start of every task:

1. Read the root `AGENTS.md`.
2. For every file you will inspect, edit, create, move, or delete, also read any `AGENTS.md` files on the path
   from the repo root to that file's directory.
   - Example: for `regain/utils/foo.py`, read in order:
     - `/AGENTS.md`
     - `/regain/AGENTS.md`, if present
     - `/regain/utils/AGENTS.md`, if present
3. Apply instructions in order from shallowest to deepest.
4. When instructions conflict, the deeper `AGENTS.md` applies for files under its directory.
5. For tasks that do not yet involve specific files, read at least the root `AGENTS.md`.
