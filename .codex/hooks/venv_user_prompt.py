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


def toml_quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


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


def is_absolute_like(path_str: str) -> bool:
    if not path_str:
        return False
    if os.path.isabs(path_str):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", path_str):
        return True
    if path_str.startswith("\\\\"):
        return True
    return False


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


def build_effective_path(bin_dir: str, current_path: str) -> str:
    parts = [bin_dir]
    if current_path:
        parts.extend(p for p in current_path.split(os.pathsep) if p)

    seen: set[str] = set()
    deduped: list[str] = []

    for part in parts:
        key = os.path.normcase(os.path.normpath(part))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)

    return os.pathsep.join(deduped)


def parse_inline_map(map_text: str) -> tuple[dict[str, str], list[str]]:
    pairs: dict[str, str] = {}
    order: list[str] = []

    for key, raw_value in re.findall(
        r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"',
        map_text,
        flags=re.MULTILINE,
    ):
        if key not in pairs:
            order.append(key)
        pairs[key] = unescape_toml_basic_string(raw_value)

    return pairs, order


def render_inline_map(values: dict[str, str], order: list[str]) -> str:
    keys = [key for key in order if key in values]
    keys.extend(key for key in values if key not in keys)
    items = [f"{key} = {toml_quote(values[key])}" for key in keys]
    return "{ " + ", ".join(items) + " }"


def find_shell_env_section(text: str) -> re.Match[str] | None:
    return re.search(
        r"(?ms)^(\[shell_environment_policy\]\s*\n)(.*?)(?=^\[|\Z)",
        text,
    )


def find_set_assignment_span(section_body: str) -> tuple[int, int, str] | None:
    line_match = re.search(r"(?m)^([ \t]*)set[ \t]*=[ \t]*\{", section_body)
    if not line_match:
        return None

    indent = line_match.group(1)
    start = line_match.start()
    brace_index = section_body.find("{", line_match.start())
    if brace_index < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    end_brace_index = None

    for index in range(brace_index, len(section_body)):
        char = section_body[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_brace_index = index
                break

    if end_brace_index is None:
        return None

    line_end = section_body.find("\n", end_brace_index)
    if line_end == -1:
        line_end = len(section_body)
    else:
        line_end += 1

    return start, line_end, indent


def ensure_features_block(text: str) -> str:
    if re.search(r"(?m)^\[features\]\s*$", text):
        if re.search(r"(?m)^\s*codex_hooks\s*=", text):
            return text

        match = re.search(r"(?ms)^(\[features\]\s*\n)(.*?)(?=^\[|\Z)", text)
        if match:
            body = match.group(2)
            suffix = "" if body.endswith("\n") or not body else "\n"
            new_body = body + suffix + "codex_hooks = true\n"
            return text[: match.start()] + match.group(1) + new_body + text[match.end() :]

    block = "[features]\ncodex_hooks = true\n\n"
    return block + text if text.strip() else block


def upsert_shell_environment_policy(
    text: str,
    env_root: str,
    bin_dir: str,
    current_path: str,
) -> str:
    text = ensure_features_block(text)

    desired_values = {
        "VIRTUAL_ENV": env_root,
        "PATH": build_effective_path(bin_dir, current_path),
    }

    match = find_shell_env_section(text)
    if not match:
        block = (
            "[shell_environment_policy]\n"
            'inherit = "all"\n'
            f"set = {render_inline_map(desired_values, list(desired_values.keys()))}\n"
        )
        return text.rstrip() + "\n\n" + block

    header = match.group(1)
    body = match.group(2)

    set_span = find_set_assignment_span(body)
    if set_span:
        set_start, set_end, indent = set_span
        existing_set_text = body[set_start:set_end]
        map_match = re.search(r"\{.*\}", existing_set_text, flags=re.DOTALL)

        existing_values: dict[str, str] = {}
        existing_order: list[str] = []
        if map_match:
            existing_values, existing_order = parse_inline_map(map_match.group(0))

        existing_values.update(desired_values)
        render_order = existing_order + [
            key for key in desired_values.keys() if key not in existing_order
        ]
        new_set_line = f"{indent}set = {render_inline_map(existing_values, render_order)}\n"
        new_body = body[:set_start] + new_set_line + body[set_end:]
    else:
        suffix = "" if body.endswith("\n") or not body else "\n"
        new_body = (
            body
            + suffix
            + f"set = {render_inline_map(desired_values, list(desired_values.keys()))}\n"
        )

    if not re.search(r'(?m)^[ \t]*inherit[ \t]*=', new_body):
        new_body = 'inherit = "all"\n' + new_body

    return text[: match.start()] + header + new_body + text[match.end() :]


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

    current_info, current_debug = inspect_virtualenv_debug(env_path)
    if current_info:
        emit(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": build_additional_context(current_info),
                }
            }
        )
        return 0

    candidate_raw = payload.get("prompt", "")
    candidate = normalize_candidate_path(candidate_raw)

    if is_absolute_like(candidate):
        info, debug = inspect_virtualenv_debug(candidate)
        if info:
            existing_text = (
                config_path.read_text(encoding="utf-8", errors="replace")
                if config_path.exists()
                else ""
            )

            updated_text = upsert_shell_environment_policy(
                existing_text,
                env_root=info["root"],
                bin_dir=info["bin_dir"],
                current_path=os.environ.get("PATH", ""),
            )

            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(updated_text, encoding="utf-8")

            emit(
                {
                    "decision": "block",
                    "reason": (
                        f"Saved Python environment to {config_path}: {info['root']}. "
                        "Send your actual task in the next prompt."
                    ),
                }
            )
            return 0

        emit(
            {
                "decision": "block",
                "reason": f"Candidate path was rejected. Debug: {debug}",
            }
        )
        return 0

    emit(
        {
            "decision": "block",
            "reason": (
                "Python environment is not configured or is invalid. "
                "Reply only with the absolute path to the environment directory "
                "or to the Python executable inside that environment. "
                f"Current config debug: {current_debug}"
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
