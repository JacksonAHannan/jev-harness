"""Start one configured OMP worker without shell interpretation or auto-approval."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .router import Agent, HarnessError, approved_model


def command_for(agent: Agent, task: str, cwd: Path, batch: bool) -> list[str]:
    if not approved_model(agent.model):
        raise HarnessError("Claude/Anthropic workers are excluded")
    if not cwd.is_dir():
        raise HarnessError("Working directory does not exist")
    if not task.strip():
        raise HarnessError("Task cannot be empty")
    command = [
        "omp",
        "--cwd",
        str(cwd.resolve()),
        "--allow-home",
        "--model",
        agent.model,
        "--no-title",
        "--no-prewalk",
        "--no-extensions",
        "--tools",
        ",".join(agent.tools),
    ]
    if not agent.tools:
        command.append("--no-tools")
    if agent.thinking:
        command.extend(["--thinking", agent.thinking])
    if agent.prompt:
        command.extend(["--append-system-prompt", agent.prompt])
    if batch:
        command.append("--print")
    # End option parsing; leading '-' or '@' must remain literal task text.
    # Prefix a heading because OMP interprets positional @file arguments as files.
    command.extend(["--", "User task:\n" + task])
    return command


def launch(agent: Agent, task: str, cwd: Path, batch: bool = False) -> int:
    command = command_for(agent, task, cwd, batch)
    env = os.environ.copy()
    # The worker uses existing OMP auth. Do not forward the router's Jev key.
    for key in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        env.pop(key, None)
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode
