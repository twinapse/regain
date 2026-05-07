# Coding agent setup

This repository includes optional project-local setup for the coding agents we use in development: Codex and Claude
Code.

<!-- toc -->

- [Setup checklist](#setup-checklist)
- [Codex](#codex)
- [Claude Code](#claude-code)

## Setup checklist

* Create or identify the Python virtual environment you want the agents to use.
* For Codex, create your local, gitignored `.codex/config.toml` and configure it for your environment
  (check the example below).
* For Claude Code, create your local, gitignored `.claude/settings.local.json` and keep your personal permissions and
  environment bootstrap logic there (check the example below).

Platform notes:

* The Python-environment validation hooks recognize both POSIX-style virtual environments (`bin/python`) and
  Windows-style ones (`Scripts/python.exe`).
* The launcher commands and examples in this document use POSIX shell conventions such as `python3`, `source`,
  `/bin`, and `:`-separated `PATH` values.
* In practice, the out-of-the-box setup is currently aimed at Linux, macOS, and WSL. Native Windows shells can use the
  same overall workflow, but you need equivalent local command and path adjustments.

## Codex

The [`.codex`](../.codex) directory contains the Codex workflow for this repo:

* [`.codex/hooks.json`](../.codex/hooks.json) registers the project hooks.
* [`.codex/hooks/venv_session_start.py`](../.codex/hooks/venv_session_start.py) checks whether a valid Python virtual
  environment is configured and adds that information as session context.
* [`.codex/hooks/venv_user_prompt.py`](../.codex/hooks/venv_user_prompt.py) blocks work until the environment is
  configured. If needed, it can accept an absolute path to a virtual environment or Python executable and write a local
  `.codex/config.toml`.

Your local `.codex/config.toml` should stay uncommitted. In this repo it is used to enable Codex hooks and to populate
`shell_environment_policy` so subprocesses inherit the same `VIRTUAL_ENV` and `PATH` values as your project shell.
Because `PATH` is stored literally, prefer letting the hook generate this file for you.
If you write it by hand, `<EXISTING_PATH_ENTRIES>` means the rest of the `PATH` entries you want to keep after the
virtual environment's `bin` directory, written as literal text rather than a shell expression such as `$PATH`.

Example `.codex/config.toml` with POSIX-style placeholders:

```toml
[features]
codex_hooks = true

[shell_environment_policy]
inherit = "all"
set = { VIRTUAL_ENV = "<ABSOLUTE_PATH_TO_VENV>", PATH = "<ABSOLUTE_PATH_TO_VENV>/bin:<EXISTING_PATH_ENTRIES>" }
```

On native Windows, keep `VIRTUAL_ENV` pointed at the environment root but switch the `PATH` entry to the environment's
`Scripts` directory and use `;` as the path separator.

Official references:

* [Codex config basics](https://developers.openai.com/codex/config-basic)
* [Codex configuration reference](https://developers.openai.com/codex/config-reference)
* [Codex hooks](https://developers.openai.com/codex/hooks)

## Claude Code

The [`.claude`](../.claude) directory is reserved for Claude Code project settings. In this repo, the relevant file is
the local, gitignored `.claude/settings.local.json`.

That local file is the right place for personal, machine-specific Claude Code setup, for example:

* `permissions.allow` rules for repetitive local commands you want to pre-approve on your own machine
* a `SessionStart` hook that prepares the same Python environment for later Bash commands via `CLAUDE_ENV_FILE`

Keep those settings local and replace any path placeholders with values from your own machine. If you later want to add
shared Claude Code settings for the whole team, put those in `.claude/settings.json` instead of the local override file.

Permissions vary by developer, so the example below only shows the shared environment hook pattern.

Example `.claude/settings.local.json` with POSIX-style placeholders:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "if [ -n \"$CLAUDE_ENV_FILE\" ] && [ -f \"<ABSOLUTE_PATH_TO_VENV>/bin/activate\" ]; then echo '[ \"$VIRTUAL_ENV\" = \"<ABSOLUTE_PATH_TO_VENV>\" ] || source \"<ABSOLUTE_PATH_TO_VENV>/bin/activate\"' >> \"$CLAUDE_ENV_FILE\"; fi"
          }
        ]
      }
    ]
  }
}
```

This `command` example assumes a POSIX shell. On native Windows, replace it with the equivalent local shell command for
your environment manager and shell.

Official references:

* [Claude Code settings](https://code.claude.com/docs/en/settings)
* [Claude Code hooks](https://code.claude.com/docs/en/hooks)
