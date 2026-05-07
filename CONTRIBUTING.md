# Contributing guidelines

This document is the source of truth for repository-wide contributor policy. It applies to human contributors and AI
agents. Agent-only workflow instructions live in [AGENTS.md](AGENTS.md).

Project reference documentation is summarized in [Documentation](#documentation).

<!-- toc -->

- [Documentation](#documentation)
- [Requirements](#requirements)
- [Coding style](#coding-style)
- [Tests and checks](#tests-and-checks)
- [Git](#git)

## Documentation

Project reference documentation lives in [docs](docs):

- [docs/analysis.md](docs/analysis.md): Repairability-frontier artifact contract and analysis output guide.
- [docs/architecture.md](docs/architecture.md): Repository architecture patterns and implementation guidance.
- [docs/coding-agent-setup.md](docs/coding-agent-setup.md): Project-local coding-agent setup guidance.
- [docs/concepts.md](docs/concepts.md): Core concepts.
- [docs/debugging.md](docs/debugging.md): Repair-controller debug instrumentation and health metric guidance.
- [docs/experimental-framework.md](docs/experimental-framework.md): Benchmark protocol, controller regimes,
  evaluation behavior, and metric logging organization.
- [docs/methods.md](docs/methods.md): Implementation notes for methods used in the codebase.
- [docs/overview.md](docs/overview.md): Research, domain, and technology-stack context.
- [docs/protocol.md](docs/protocol.md): Research protocol for repairability-frontier construction and repair-selection evaluation.

Review the relevant reference documentation before changing related behavior.

## Requirements

This project is tested on Python 3.11 in CI. Runtime dependencies are listed in [requirements.txt](requirements.txt),
and development dependencies are listed in [requirements-dev.txt](requirements-dev.txt). CI uses
[requirements-ci-tests.txt](requirements-ci-tests.txt) for a stable test dependency set.

For local development, install the runtime and development dependencies in the active project environment:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Coding style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and the
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). Check [pylintrc](pylintrc)
and [.style.yapf](.style.yapf) for the formatter and linter configuration that backs these rules.

To ensure code consistency, use the following tools:

1. [isort](https://pycqa.github.io/isort/): organizes imports.

   ```bash
   isort regain tests --profile google --line-length 120
   ```

2. [YAPF](https://github.com/google/yapf): formats Python source files.

   ```bash
   yapf --recursive --in-place regain tests
   ```

3. [Ruff](https://github.com/astral-sh/ruff): runs fast lint checks and auto-fixes supported issues.

   ```bash
   ruff check regain tests --fix
   ```

4. [Pylint](https://pypi.org/project/pylint/): runs the stricter lint pass used by this project.

   ```bash
   pylint --rcfile=pylintrc regain
   ```

   For tests, use the Pytest-specific exceptions already established in the repo:

   ```bash
   pylint --rcfile=pylintrc --disable=redefined-outer-name,unused-argument tests
   ```

### Formatting

- Use 4-space indentation.
- Keep lines at 120 characters or fewer.
- Use single quotes for string literals by default.
- Use double quotes only when it avoids awkward escaping.
- Prefer `f`-strings over `%` formatting or `.format(...)`.
- Break dense function signatures and calls across multiple lines.

Example:

```python
metric_key = 'analysis.repair.rho.avg'
error = f'Unsupported controller: {name}'
label = "Controller's calibration"
```

Avoid:

```python
metric_key = "analysis.repair.rho.avg"
error = 'Unsupported controller: {}'.format(name)
label = 'Controller\'s calibration'
```

When a call or expression becomes dense, break it across lines clearly and consistently:

```python
parser.add_argument(
    '--tracking-uri',
    type=str,
    default=None,
    help='Optional MLflow tracking URI override.',
)
```

Avoid:

```python
parser.add_argument('--tracking-uri', type=str, default=None, help='Optional MLflow tracking URI override.')
```

### Typing and arguments

- All functions and methods must include type hints, including return types.
- Prefer keyword arguments for multi-parameter calls and public APIs.
- Use keyword-only parameters (`*`) when named call sites improve readability or safety.
- Single obvious parameters may stay positional.
- Preserve framework-prescribed signatures when external integrations depend on them.

Use type hints on all helpers, and choose positional versus keyword-only parameters based on call-site clarity:

```python
def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    ...


def cast_tensor(*, tensor: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    ...
```

Avoid:

```python
def load_experiment_config(config_path):
    ...


def cast_tensor(tensor, ref_tensor):
    ...
```

Prefer keywords for dense calls:

```python
targets = resolve_experiment_targets(
    parser=parser,
    config_files=args.config_files,
    config_dir=args.config_dir,
    experiments=args.experiments,
    tracking_uri=args.tracking_uri,
    failures=failures,
)
```

Single obvious arguments may remain positional:

```python
experiment_config = load_experiment_config(config_path)
```

Framework-specific note:

- Preserve signatures that are dictated by the surrounding framework or public interface contract. Do not reshape those
  interfaces just to satisfy a local preference for keyword-only or positional arguments.

### Docstrings and comments

- Use Google-style docstrings with `Args`, `Returns`, and `Raises` when relevant.
- Use the newline-after-opening-quotes style: the opening `"""` is on its own line and the summary starts on the next
  line.
- Include type information in docstring sections.
- Indent docstring field descriptions after the section labels.
- Use triple double quotes only for docstrings.
- Use `#` comments for inline notes and lightweight banner comments for major sections.

Example:

```python
def get_backbone_path(name: str) -> str:
    """
    Resolve a backbone name to its fully qualified path.

    Args:
        name (str): Backbone registry name.

    Returns:
        str: Fully qualified path for the backbone class.

    Raises:
        ValueError: If the backbone name is invalid or unsupported.
    """
```

Avoid same-line summaries and block-comment docstrings:

```python
def get_backbone_path(name: str) -> str:
    """Resolve a backbone name to its fully qualified path."""


"""
Export generation
"""
```

Use `#` comments or banner comments instead:

```python
#####################
# Export generation #
#####################

# Serialize the final analysis bundle after all CSV sections are loaded.
```

### Imports and exports

- Do not use `from __future__ ...` imports.
- Use absolute imports; avoid relative imports.
- Group imports as stdlib, third-party, then project imports.
- Keep imports alphabetized within each group.
- Import one symbol per line, except for `typing` imports.
- Declare `__all__` immediately after imports in modules that export a public surface.
- Keep one line per exported symbol in `__all__`.
- Keep `__all__` in sync when adding or removing public exports.

Example:

```python
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from regain.constants import PARAM_CONTROLLER
from regain.constants import PARAM_SCENARIO
from regain.registry import get_backbone_path

__all__ = [
    'ExperimentConfig',
    'load_experiment_config',
]
```

Avoid:

```python
from __future__ import annotations
from .registry import get_backbone_path
from regain.constants import PARAM_CONTROLLER, PARAM_SCENARIO
```

When a module exposes helpers publicly, keep `__all__` explicit and aligned with the file contents:

```python
__all__ = [
    'ExperimentTarget',
    'add_experiment_selector_arguments',
    'find_config_files',
    'resolve_experiment_targets',
]
```

### Naming

- Use `snake_case` for functions, variables, and modules.
- Use `PascalCase` for classes and enums.
- Use `ALL_CAPS` for constants.
- Keep names aligned with the existing domain language in the repository.

Repository-shaped examples:

```python
def load_experiment_config(config_path: str) -> ExperimentConfig:
    ...


class RepairConfig:
    ...


RUN_RHO_AVG = 'run.repair.rho.avg'
```

## Tests and checks

Run the relevant tests before opening or updating a pull request:

```bash
pytest
```

Use focused test paths while iterating:

```bash
pytest tests/test_cli_run_analysis.py
```

The CI workflow in [.github/workflows/tests.yml](.github/workflows/tests.yml) runs:

```bash
python -m pytest -q tests -ra --maxfail=1
```

## Git

A clean Git history makes it easier to understand why a change was made and what behavior it affects.

### Commits

Use the following commit message format:

```text
<type>(<scope>): <short summary>

<summary>

Changes:
- <change 1>
- <change 2>
```

1. **Type**
   - A brief label describing the kind of change:
     - `feat`: new feature
     - `fix`: bug fix
     - `docs`: documentation-only change
     - `style`: non-behavioral formatting or style-only change
     - `refactor`: code restructuring without behavior change
     - `perf`: performance improvement
     - `test`: test-only change
     - `chore`: maintenance or tooling update

2. **Scope (optional)**
   - Use a narrow, meaningful scope such as `analysis`, `config`, `repair`, `backbone`, or `cli`.

3. **Short summary**
   - Keep it under 50 characters.
   - Use imperative mood.
   - Do not end it with a period.

4. **Summary**
   - Explain what changed and why.

5. **Changes**
   - Add a short bullet list for non-trivial commits.
   - Start each bullet item with a capital letter.

Example commit message:

```text
feat(analysis): export analysis bundles

Write a self-contained `analysis.json` bundle so downstream tooling can
consume tables and derived outputs from a single artifact.

Changes:
- Add bundle assembly in `regain.analysis.exports`
- Expose the export through `regain.cli.export_analysis`
- Cover the CLI contract in `tests/test_cli_export_analysis.py`
```

Also make sure to:

- Use imperative mood: `add`, `fix`, `remove`, `update`.
- Be specific about the affected subsystem or behavior.
- Separate the subject from the body with a blank line.
- Reference related issues in the footer when relevant.
- Keep each commit focused on one logical change.
- Use backticks for code references such as function names, class names, and file paths.

### Pull requests

PR titles and descriptions should make the change understandable without opening the diff first.

#### Title

Format:

```text
[Scope] Short, descriptive title
```

Example:

```text
[Analysis] Export Analysis Bundles
```

Guidelines:

1. Keep the title concise.
2. Use imperative mood.
3. Avoid generic titles such as `Fix bug` or `Update code`.
4. Reference an issue or ticket when that context matters.
5. Avoid unnecessary punctuation or decoration.

#### Description

Include:

- A `Summary` section that starts with a short paragraph and may be followed by bullets
- A `Motivation` section that starts with a short paragraph and may be followed by bullets
- Important context or constraints
- A categorized list of new features, breaking changes, behavior changes, refactors, or fixes
- The tests that were added or updated to cover the change
- Bullet items that start with a capital letter when bullets are used
- Code references, including function names, class names, commands, and file paths, enclosed with backticks

Example PR description:

```markdown
## Summary

This change exports self-contained analysis bundles for downstream tooling.

## Motivation

Downstream consumers should not need to reload multiple intermediate CSV files to inspect analysis outputs.

## Behavior changes

- Write a self-contained `analysis.json` bundle for downstream consumers
- Include exported tables and derived outputs in a single artifact

## Refactors

- Extend `regain.analysis.exports` to serialize the bundle
- Update `regain.cli.export_analysis` to write the export artifact

## Tests

- Run `pytest tests/test_cli_export_analysis.py`
- Add coverage for the export contract in `tests/test_cli_export_analysis.py`
```
