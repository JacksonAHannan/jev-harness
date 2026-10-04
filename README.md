# Jev Harness

A Python CLI that routes a task to an existing Oh My Pi agent role using TypeSafe's Jev choice API. The harness validates the routing response, applies configurable confidence thresholds and launches one worker with that role's configured model, instructions and tools.

```text
Task → available role catalog → Jev Choice → validated decision → OMP worker
                                           ↓ low confidence
                                      configured fallback
```

## What it demonstrates

- Integration with an installed SDK and agent configuration, including model overrides and disabled roles.
- Validation of finite, complete probability distributions and allowed destinations before dispatch.
- Explicit fallback policy for uncertain choices and fail-closed behavior for malformed responses/API errors.
- Argument-safe process launch: task text is passed as one argument, including shell-looking text.
- Credential isolation and optional private decision receipts that contain a task hash rather than task text.

Routing happens once before the worker session. The configured roles are `smol`, `task`, `default`, `plan`, `slow` and `vision`; only locally available roles are offered. This implementation excludes Claude/Anthropic destinations, including aggregated model IDs. The plan role retains read-only tools. Worker approvals remain governed by the existing OMP installation.

## Setup and use

Requires Python 3.11+, `uv`, an existing authenticated Oh My Pi (`omp`) installation and its role definitions/configuration under `~/.omp/agent/`. A TypeSafe routing API key is required for `route` and `run`. Existing role configuration and worker credentials are managed by OMP, outside this repository.

```sh
uv sync --frozen
cp .env.example .env
chmod 600 .env
```

Set your own key in the local `.env`. The launcher loads that file only when neither `TYPESAFE_API_KEY` nor `JEV_API_KEY` is already set. The template contains a placeholder; no credentials are bundled. To inspect the local role catalog without invoking Jev:

```sh
./bin/jev-harness agents
```

From the project where you want the worker to run, use the absolute path to this checkout's launcher:

```sh
/path/to/jev-harness/bin/jev-harness route "Find where CLI arguments are parsed"
/path/to/jev-harness/bin/jev-harness run "Implement the CSV export and verify it"
/path/to/jev-harness/bin/jev-harness run "Create an implementation plan; do not edit files"
/path/to/jev-harness/bin/jev-harness run "Find the entrypoint" --batch
```

`route` invokes the routing API but does not start a worker. `run` invokes the API and starts an OMP session. Use `--cwd /path/to/project`, `--task-file /path/to/task.md` or `--receipt /private/decisions.jsonl` when needed. Global `--config` and `--omp-dir` arguments precede the command. `--help` documents the full interface.

## Routing policy and records

[router.toml](src/jev_harness/router.toml) defines role descriptions and initial thresholds: confidence ≥0.70 and selected-option probability ≥0.65. Lower values select the configured `default` fallback. These are policy choices rather than empirically calibrated cutoffs. Missing keys, unknown destinations, non-finite values and API failures stop routing before worker launch. An unavailable fallback is an error.

Every invocation rereads OMP's agent Markdown and YAML configuration. The harness removes the routing credential from the worker environment. It disables task spawning, automatic prewalk model switching, extension discovery and title generation for the routed session. Receipts are optional, owner-private JSONL containing the task digest, selected/proposed role, model, probabilities, timestamp, working directory, token usage and worker exit status. Keep them outside public repositories.

## Development and verification

```sh
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
bash -n bin/jev-harness
uv build
```

The 21-test offline suite mocks the SDK HTTP transport and dispatches to a stand-in executable. Tests cover response validation, confidence fallbacks, role overrides, credential isolation and argument safety. They do not invoke a paid API or worker model. [Verification](docs/VERIFICATION.md) records checks actually run on this snapshot.

This is a local integration tool, dependent on installed SDK/OMP behavior. No production routing-accuracy or performance claim is made. No software license grant has been supplied; copyright remains with the respective owners.

API references: [TypeSafe intent routing](https://docs.typesafe.ai/patterns/intent-routing), [Choice](https://docs.typesafe.ai/primitives/choice), [confidence routing](https://docs.typesafe.ai/patterns/confidence-routing), [Python SDK](https://docs.typesafe.ai/sdk/python).
