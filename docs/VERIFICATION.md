# Publication verification — 2026-10-03

Checks run on the isolated source snapshot:

| Check | Result |
| --- | --- |
| Pytest against the snapshot source | 21 tests passed, 0.90 seconds, exit 0 |
| Ruff `check --no-cache .` | Passed, exit 0 |
| Ruff `format --check --no-cache .` | Six Python files already formatted, exit 0 |
| `bash -n bin/jev-harness` | Passed, exit 0 |
| `uv build --offline` | Source distribution and wheel built, exit 0 |

Pytest used the original project's already-installed Python/dependencies with `PYTHONPATH=src` and `PYTHONDONTWRITEBYTECODE=1`, plus `-p no:cacheprovider`. Ruff used the same pre-existing environment. No software installation, real routing API request or live worker model was needed. On a fresh development checkout, `uv sync --frozen` followed by `uv run --frozen pytest -q` is the documented equivalent.

The SDK HTTP transport is mocked and worker dispatch uses a fixture executable. This verifies routing contracts and process arguments; it does not establish production model accuracy or end-to-end live integration. Wheel ZIP inspection confirmed the default `jev_harness/router.toml` is packaged. Application source, configuration and tests are unchanged from the local original.
