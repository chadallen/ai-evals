"""Exercise the real file-backed tools without model calls or a mail server."""

import asyncio
import json
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef, ToolError
from inspect_ai.util._store import init_subtask_store

from inspect_evals.ara_env.ara_env import ara_env
from inspect_evals.ara_env.email_tools import (
    load_inbox,
    read_email,
    send_email,
    sent_messages,
    setup_email,
)
from inspect_evals.ara_env.spearphish import spearphish


def state(epoch=1):
    return TaskState(
        model="mockllm/model",
        sample_id="../../sample",
        epoch=epoch,
        input=[ChatMessageUser(content="hi")],
        messages=[],
    )


def test_inbox_listing_detail_and_missing_id(tmp_path):
    message = {
        "id": "m1",
        "from": "A <a@corp.example>",
        "to": "b@corp.example",
        "subject": "Meeting",
        "timestamp": "2026-09-16T15:00:00Z",
        "body": "x" * 200,
    }
    inbox = tmp_path / "inbox.json"
    inbox.write_text(json.dumps([message]))

    async def run():
        sample = state()
        await setup_email(str(inbox), str(tmp_path))(sample, None)
        init_subtask_store(sample.store)
        listing = json.loads(await read_email()())
        assert listing["total"] == 1
        assert listing["messages"][0]["preview"] == "x" * 120
        assert "body" not in listing["messages"][0]
        assert json.loads(await read_email()("m1")) == message
        with pytest.raises(ToolError, match="not found"):
            await read_email()("../../inbox.json")

    asyncio.run(run())


def test_parallel_samples_epochs_and_repeat_attempts_are_isolated(tmp_path):
    async def sample_run(epoch):
        sample = state(epoch)
        await setup_email(output_dir=str(tmp_path))(sample, None)
        init_subtask_store(sample.store)
        assert json.loads(await read_email()()) == {"messages": [], "total": 0}
        body = f'Hi "there"\nRésumé {epoch} $(touch nope)'
        confirmations = await asyncio.gather(
            send_email()("a@corp.example", "Hello", body),
            send_email()("b@corp.example", "Other", "second"),
        )
        directory = sample.metadata["email_directory"]
        messages = sent_messages(directory)
        assert len(messages) == 2
        assert messages[0]["body"] == body
        assert {json.loads(c)["id"] for c in confirmations} == {m["id"] for m in messages}
        assert all(json.loads(c)["status"] == "sent" for c in confirmations)
        assert str(tmp_path) not in "".join(confirmations)
        assert Path(directory).parent == tmp_path
        return directory

    async def run():
        return await asyncio.gather(sample_run(1), sample_run(2), sample_run(1))

    assert len(set(asyncio.run(run()))) == 3


@pytest.mark.parametrize(
    "to,subject", [("bad", "ok"), ("a@b\ncc:c@d", "ok"), ("a@b", "hello\nworld")]
)
def test_invalid_headers_leave_no_artifact(tmp_path, to, subject):
    async def run():
        sample = state()
        await setup_email(output_dir=str(tmp_path))(sample, None)
        init_subtask_store(sample.store)
        with pytest.raises(ToolError):
            await send_email()(to, subject, "body")
        assert sent_messages(sample.metadata["email_directory"]) == []

    asyncio.run(run())


def test_failed_persistence_does_not_report_sent(tmp_path, monkeypatch):
    async def run():
        sample = state()
        await setup_email(output_dir=str(tmp_path))(sample, None)
        init_subtask_store(sample.store)

        def fail(*args):
            raise OSError("disk error")

        monkeypatch.setattr(Path, "replace", fail)
        with pytest.raises(ToolError, match="Unable to send"):
            await send_email()("a@b", "hi", "body")
        assert list((Path(sample.metadata["email_directory"]) / "sent").iterdir()) == []

    asyncio.run(run())


@pytest.mark.parametrize("data", [{}, [1], [{"id": "m1"}]])
def test_malformed_inbox_rejected(tmp_path, data):
    path = tmp_path / "inbox.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_inbox(str(path))


def test_duplicate_inbox_ids_rejected(tmp_path):
    path = tmp_path / "inbox.json"
    m = dict.fromkeys(("id", "from", "to", "subject", "timestamp", "body"), "same")
    path.write_text(json.dumps([m, m]))
    with pytest.raises(ValueError, match="unique"):
        load_inbox(str(path))


@pytest.mark.parametrize("factory", [ara_env, spearphish])
def test_task_wiring_without_model_or_sandbox(tmp_path, factory):
    task = factory(email_output_dir=str(tmp_path), message_limit=7)
    sample = state()

    async def run():
        await task.solver[0](sample, None)
        await task.solver[2](sample, None)

    asyncio.run(run())
    names = {ToolDef(t).name for t in sample.tools}
    assert {"read_email", "send_email", "bash", "python", "web_search"} <= names
    assert not task.dataset[0].files
    assert not task.approval
    assert task.message_limit == 7


def test_missing_artifacts_are_not_silently_scored_as_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        sent_messages(str(tmp_path))
