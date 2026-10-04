"""Command line entry point for inspecting routes or starting OMP sessions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .dispatch import launch
from .router import HarnessError, ask_jev, decide, load_catalog, write_receipt

DEFAULT_CONFIG = Path(__file__).with_name("router.toml")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Route tasks with Jev to existing Oh My Pi roles")
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    root.add_argument("--omp-dir", type=Path, default=Path.home() / ".omp" / "agent")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("agents", help="Show available agent roles and excluded destinations")
    for name in ("route", "run"):
        command = commands.add_parser(
            name, help="Inspect routing" if name == "route" else "Route and start one OMP worker"
        )
        command.add_argument("task", nargs="?", help="Task text; use --task-file for long tasks")
        command.add_argument("--task-file", type=Path, help="Read task from a UTF-8 file")
        command.add_argument("--cwd", type=Path, default=Path.cwd())
        command.add_argument("--receipt", type=Path, help="Append private JSONL decision records")
        if name == "run":
            command.add_argument("--batch", action="store_true", help="Use OMP print mode")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        catalog = load_catalog(args.config.expanduser(), args.omp_dir.expanduser())
        if args.command == "agents":
            print(
                json.dumps(
                    {
                        "agents": {
                            role: {
                                "model": agent.model,
                                "thinking": agent.thinking,
                                "description": agent.description,
                                "tools": agent.tools,
                            }
                            for role, agent in catalog.agents.items()
                        },
                        "excluded": catalog.excluded,
                    },
                    indent=2,
                )
            )
            return 0
        if (args.task is None) == (args.task_file is None):
            raise HarnessError("Provide either task text or --task-file")
        task = args.task_file.expanduser().read_text() if args.task_file else args.task
        if not task.strip():
            raise HarnessError("Task cannot be empty")
        cwd = args.cwd.expanduser().resolve()
        if not cwd.is_dir():
            raise HarnessError("Working directory does not exist")
        api_key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not api_key:
            raise HarnessError("Set TYPESAFE_API_KEY or JEV_API_KEY in your environment")
        decision = decide(task, catalog, ask_jev(task, catalog, api_key))
        record = {**decision.to_dict(), "event": "route", "cwd": str(cwd)}
        if args.receipt:
            write_receipt(args.receipt.expanduser(), record)
        if args.command == "route":
            print(json.dumps(record, indent=2))
            return 0
        print(
            f"Jev → {decision.agent} ({decision.model}); confidence {decision.confidence:.3f}"
            + ("; fallback" if decision.fallback else ""),
            file=sys.stderr,
        )
        status = launch(catalog.agents[decision.agent], task, cwd, args.batch)
        if args.receipt:
            write_receipt(
                args.receipt.expanduser(),
                {
                    "event": "worker_exit",
                    "id": decision.id,
                    "exit_code": status,
                },
            )
        return status if status >= 0 else 128 - status
    except KeyboardInterrupt:
        return 130
    except HarnessError as error:
        print(f"jev-harness: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        # SDK/OS exceptions may include request bodies, headers, or paths to secrets.
        # Print only the class, never the remote response or raw exception text.
        print(f"jev-harness: {type(error).__name__}; routing stopped", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
