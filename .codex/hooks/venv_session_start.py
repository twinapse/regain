#!/usr/bin/env python3

import json
import os
import pathlib
import re
import sys


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def unescape_toml_basic_string(value: str) -> str:
    return bytes(value, "utf-8").decode("unicode_escape")


def find_project_root(start_cwd: str) -> pathlib.Path:
    current = pathlib.Path(start_cwd).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".codex" / "config.toml").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return current


def read_configured_virtual_env(config_path: pathlib.Path) -> str:
    if not config_path.exists():
        return ""

    text = config_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r'VIRTUAL_ENV\s*=\s*"((?:[^"\\]|\\.)*)"',
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return ""

    return unescape_toml_basic_string(match.group(1)).strip()


def normalize_candidate_path(raw_path: str) -> str:
    value = raw_path.strip()
    if not value:
        return ""

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1].strip()

    return os.path.expandvars(os.path.expanduser(value))


def derive_env_root(candidate_path: str) -> pathlib.Path:
    path = pathlib.Path(candidate_path)

    name_lower = path.name.lower()
    parent_name_lower = path.parent.name.lower()

    python_names = {"python", "python.exe"}
    activate_names = {"activate", "activate.bat", "activate.ps1"}

    if name_lower in python_names | activate_names and parent_name_lower in {"bin", "scripts"}:
        return path.parent.parent

    return path


def inspect_virtualenv_debug(raw_path: str) -> tuple[dict[str, str] | None, str]:
    normalized = normalize_candidate_path(raw_path)
    if not normalized:
        return None, "empty path"

    root = derive_env_root(normalized)

    posix = {
        "bin_dir": root / "bin",
        "python": root / "bin" / "python",
        "pip": root / "bin" / "pip",
        "activate": root / "bin" / "activate",
    }
    windows_exe = {
        "bin_dir": root / "Scripts",
        "python": root / "Scripts" / "python.exe",
        "pip": root / "Scripts" / "pip.exe",
        "activate": root / "Scripts" / "activate.bat",
    }
    windows_noexe = {
        "bin_dir": root / "Scripts",
        "python": root / "Scripts" / "python",
        "pip": root / "Scripts" / "pip",
        "activate": root / "Scripts" / "activate",
    }

    checks = [
        ("input_exists", pathlib.Path(normalized).exists()),
        ("input_is_dir", pathlib.Path(normalized).is_dir()),
        ("derived_root", str(root)),
        ("root_exists", root.exists()),
        ("root_is_dir", root.is_dir()),
        ("bin_python", posix["python"].exists()),
        ("bin_pip", posix["pip"].exists()),
        ("bin_activate", posix["activate"].exists()),
        ("scripts_python_exe", windows_exe["python"].exists()),
        ("scripts_pip_exe", windows_exe["pip"].exists()),
        ("scripts_activate_bat", windows_exe["activate"].exists()),
        ("scripts_python", windows_noexe["python"].exists()),
        ("scripts_pip", windows_noexe["pip"].exists()),
        ("scripts_activate", windows_noexe["activate"].exists()),
    ]

    debug = (
        f"input={raw_path!r}; "
        f"normalized={normalized!r}; "
        + "; ".join(
            f"{name}={value!r}" if name == "derived_root" else f"{name}={value}"
            for name, value in checks
        )
    )

    candidates = [posix, windows_exe, windows_noexe]

    for candidate in candidates:
        if candidate["python"].exists() or candidate["activate"].exists():
            info = {
                "root": str(root),
                "bin_dir": str(candidate["bin_dir"]),
                "python": str(candidate["python"]),
                "pip": str(candidate["pip"]),
                "activate": str(candidate["activate"]),
            }
            return info, debug

    return None, debug


def build_additional_context(info: dict[str, str]) -> str:
    return (
        f"Project Python environment: {info['root']}. "
        f"Use Python at {info['python']} and pip at {info['pip']}. "
        "Assume subprocesses receive VIRTUAL_ENV and PATH from shell_environment_policy.set."
    )


def main() -> int:
    payload = json.load(sys.stdin)
    project_root = find_project_root(payload["cwd"])
    config_path = project_root / ".codex" / "config.toml"

    env_path = os.environ.get("VIRTUAL_ENV", "").strip()
    if not env_path:
        env_path = read_configured_virtual_env(config_path)

    info, debug = inspect_virtualenv_debug(env_path)

    if info:
        emit(
            {
                "systemMessage": f"Activated Python environment: {info['root']}",
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": build_additional_context(info),
                },
            }
        )
        return 0

    emit(
        {
            "systemMessage": (
                "Python environment is not configured or is invalid. "
                f"Debug: {debug}"
            ),
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    "Do not do project work until the Python environment is configured."
                ),
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
