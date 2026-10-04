from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import httpx2
import pytest
import yaml
from typesafe_sdk import TypeSafeClient

from jev_harness import cli, router
from jev_harness.dispatch import command_for, launch
from jev_harness.router import HarnessError, decide, load_catalog, write_receipt


@pytest.fixture
def setup(tmp_path):
    omp_dir = tmp_path / "omp"
    (omp_dir / "agents").mkdir(parents=True)
    config = tmp_path / "router.toml"
    config.write_text(
        'fallback_role = "default"\n[roles]\n'
        'smol = "Small tasks"\ndefault = "General work"\nplan = "Read-only plans"\n'
        'advisor = "Review"\n'
    )
    models = {
        "smol": "deepseek/deepseek-v4-flash:low",
        "default": "openai-codex/gpt-6-astra:low",
        "plan": "openai-codex/gpt-6-astra:high",
        "advisor": "openrouter/anthropic/claude-opus-5-5:high",
    }
    (omp_dir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "modelRoles": models,
                "task": {"agentModelOverrides": {"default": "openai-codex/gpt-6.1-sol"}},
            }
        )
    )
    for role in models:
        (omp_dir / "agents" / f"{role}.md").write_text(
            f'---\nname: {role}\nmodel: ["@{role}"]\nspawns: []\n---\nDo {role} work.\n'
        )
    return config, omp_dir


@pytest.fixture
def catalog(setup):
    return load_catalog(*setup)


def response(catalog, choice="smol", confidence=0.99, probability=0.99):
    others = [role for role in catalog.agents if role != choice]
    probabilities = {role: (1 - probability) / len(others) for role in others}
    probabilities[choice] = probability
    return {
        "model": "jev-1.13.0",
        "answers": {
            "agent": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 250, "output_tokens": 20},
    }


def test_catalog_resolves_overrides_and_excludes_anthropic(catalog):
    assert catalog.agents["smol"].model == "deepseek/deepseek-v4-flash"
    assert catalog.agents["smol"].thinking == "low"
    assert catalog.agents["default"].model == "openai-codex/gpt-6.1-sol"
    assert "advisor" not in catalog.agents
    assert "Claude/Anthropic" in catalog.excluded["advisor"]
    assert set(catalog.agents["plan"].tools) == {"read", "grep", "glob", "web_search"}
    assert all("task" not in agent.tools for agent in catalog.agents.values())


def test_model_changes_are_read_without_cached_catalog(setup):
    config, omp_dir = setup
    path = omp_dir / "config.yml"
    settings = yaml.safe_load(path.read_text())
    settings["modelRoles"]["smol"] = "openrouter/claude-sonnet-5"
    path.write_text(yaml.safe_dump(settings))
    assert "smol" not in load_catalog(config, omp_dir).agents
    settings["task"]["agentModelOverrides"]["default"] = "anthropic/claude-opus-5-5"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(HarnessError, match="Fallback"):
        load_catalog(config, omp_dir)


def test_missing_and_disabled_agents_are_not_offered(setup):
    config, omp_dir = setup
    (omp_dir / "agents" / "plan.md").unlink()
    path = omp_dir / "config.yml"
    settings = yaml.safe_load(path.read_text())
    settings["task"]["disabledAgents"] = ["smol"]
    path.write_text(yaml.safe_dump(settings))
    catalog = load_catalog(config, omp_dir)
    assert set(catalog.agents) == {"default"}
    assert set(catalog.excluded) == {"smol", "plan", "advisor"}


def test_high_confidence_selects_specialist(catalog):
    decision = decide("Find the parser", catalog, response(catalog))
    assert decision.agent == "smol"
    assert decision.fallback is False
    assert "Find the parser" not in json.dumps(decision.to_dict())


@pytest.mark.parametrize("confidence,probability", [(0.2, 0.99), (0.99, 0.6)])
def test_uncertain_selection_uses_default(catalog, confidence, probability):
    decision = decide(
        "Some work", catalog, response(catalog, confidence=confidence, probability=probability)
    )
    assert decision.agent == "default"
    assert decision.proposed_agent == "smol"
    assert decision.fallback is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["answers"].clear(),
        lambda r: r["answers"]["agent"].update(type="noul"),
        lambda r: r["answers"]["agent"].update(choice="advisor"),
        lambda r: r["answers"]["agent"].update(confidence=float("nan")),
        lambda r: r["answers"]["agent"].update(confidence=True),
        lambda r: r["answers"]["agent"]["probabilities"].update(smol=2),
        lambda r: r["answers"]["agent"]["probabilities"].pop("plan"),
        lambda r: r["answers"]["agent"].update(choice="plan"),
        lambda r: r["usage"].update(input_tokens=-1),
    ],
)
def test_malformed_answers_stop_routing(catalog, mutation):
    payload = response(catalog)
    mutation(payload)
    with pytest.raises(HarnessError):
        decide("Task", catalog, payload)


def test_sdk_transport_contract_and_excluded_options(catalog, monkeypatch):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=response(catalog))

    monkeypatch.setattr(
        router,
        "TypeSafeClient",
        lambda **kwargs: TypeSafeClient(**kwargs, transport=httpx2.MockTransport(handler)),
    )
    result = router.ask_jev("Find the parser", catalog, "fake-test-key")
    decision = decide("Find the parser", catalog, result)
    request = captured[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["state"] == {"task": "Find the parser"}
    assert set(body["questions"]["agent"]["criteria"]) == set(catalog.agents)
    assert "advisor" not in body["questions"]["agent"]["criteria"]
    assert decision.agent == "smol"


def test_dispatch_preserves_task_cwd_exit_and_strips_keys(catalog, tmp_path, monkeypatch):
    # Stand-in executable: no worker model is called during these checks.
    output = tmp_path / "worker.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "omp"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['WORKER_OUTPUT']).write_text(json.dumps({"
        "'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'key_present': 'TYPESAFE_API_KEY' in os.environ or 'JEV_API_KEY' in os.environ}))\n"
        "sys.exit(7)\n"
    )
    fake.chmod(0o700)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("WORKER_OUTPUT", str(output))
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-secret")
    monkeypatch.setenv("JEV_API_KEY", "fake-secret")
    task = "--auto-approve @private-file $(touch unwanted-file)"
    status = launch(catalog.agents["smol"], task, tmp_path, batch=True)
    result = json.loads(output.read_text())
    assert status == 7
    assert result["cwd"] == str(tmp_path)
    assert not result["key_present"]
    args = result["args"]
    assert args[-2:] == ["--", "User task:\n" + task]
    assert "--auto-approve" not in args
    assert "--no-prewalk" in args
    assert "--print" in args
    assert not (tmp_path / "unwanted-file").exists()


def test_dispatch_rejects_anthropic_even_after_routing(catalog, tmp_path):
    blocked = replace(catalog.agents["default"], model="openrouter/anthropic/claude-opus")
    with pytest.raises(HarnessError, match="excluded"):
        command_for(blocked, "Task", tmp_path, False)


def test_receipts_are_private_and_do_not_follow_symlinks(tmp_path):
    path = tmp_path / "receipts" / "decisions.jsonl"
    write_receipt(path, {"id": "first"})
    write_receipt(path, {"id": "second"})
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == ["first", "second"]
    assert path.stat().st_mode & 0o777 == 0o600
    link = tmp_path / "linked-receipt"
    link.symlink_to(path)
    with pytest.raises(OSError):
        write_receipt(link, {"id": "bad"})


def test_cli_api_failure_cannot_launch_or_leak_secret(setup, monkeypatch, capsys):
    config, omp_dir = setup
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-secret")

    def fail(*args):
        raise RuntimeError("Remote error containing fake-secret")

    monkeypatch.setattr(cli, "ask_jev", fail)
    monkeypatch.setattr(cli, "launch", lambda *args: pytest.fail("Worker must not launch"))
    status = cli.main(["--config", str(config), "--omp-dir", str(omp_dir), "run", "Task"])
    output = capsys.readouterr()
    assert status == 1
    assert "RuntimeError" in output.err
    assert "fake-secret" not in output.err


def test_cli_run_records_worker_exit(setup, monkeypatch, capsys, tmp_path):
    config, omp_dir = setup
    catalog = load_catalog(*setup)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-secret")
    monkeypatch.setattr(cli, "ask_jev", lambda *args: response(catalog))
    launches = []
    monkeypatch.setattr(cli, "launch", lambda *args: launches.append(args) or 3)
    receipt = tmp_path / "receipts.jsonl"
    status = cli.main(
        [
            "--config",
            str(config),
            "--omp-dir",
            str(omp_dir),
            "run",
            "Find the parser",
            "--cwd",
            str(tmp_path),
            "--receipt",
            str(receipt),
            "--batch",
        ]
    )
    assert status == 3
    assert launches[0][0].role == "smol"
    assert launches[0][1:] == ("Find the parser", tmp_path, True)
    records = [json.loads(line) for line in receipt.read_text().splitlines()]
    assert [r["event"] for r in records] == ["route", "worker_exit"]
    assert records[1]["exit_code"] == 3
    assert records[0]["id"] == records[1]["id"]
    assert "fake-secret" not in receipt.read_text()
    assert "smol" in capsys.readouterr().err
