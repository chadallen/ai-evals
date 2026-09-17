"""Exercise the real file-backed tools without model calls or a mail server."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef, ToolError, tool
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


@tool
def browser_navigate():
    async def execute(url: str) -> str:
        """Open a URL in the test browser.

        Args:
            url: Page to open.
        """
        return url

    return execute


@tool
def browser_click():
    async def execute(target: str) -> str:
        """Click an element in the test browser.

        Args:
            target: Element to click.
        """
        return target

    return execute


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
        assert json.loads(await read_email()("m1")) == {**message, "content_type": "text/plain"}
        assert listing["messages"][0]["content_type"] == "text/plain"
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
        assert all(m["content_type"] == "text/plain" for m in messages)
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
    module = factory.__module__
    browser = [browser_navigate(), browser_click()]
    with patch(f"{module}.playwright_browser_tools", return_value=browser):
        task = factory(email_output_dir=str(tmp_path), message_limit=7)
    sample = state()

    async def run():
        await task.solver[0](sample, None)
        async def no_generate(current, **kwargs):
            return current

        await task.solver[2](sample, no_generate)

    asyncio.run(run())
    names = {ToolDef(t).name for t in sample.tools}
    assert {"read_email", "send_email", "bash", "python"} <= names
    assert {"browser_navigate", "browser_click"} <= names
    if factory is ara_env:
        assert "web_search" in names
    else:
        assert "web_search" not in names
        assert not any(name.startswith("web_browser") for name in names)
    assert not task.dataset[0].files
    if factory is ara_env:
        assert not task.approval
    assert task.message_limit == 7


def test_missing_artifacts_are_not_silently_scored_as_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        sent_messages(str(tmp_path))


def test_html_inbox_and_send_preserve_body_and_type(tmp_path):
    body = '<p>Résumé &amp; news</p><a href="https://corp.example">Read more</a>'
    message = dict.fromkeys(("id", "from", "to", "subject", "timestamp"), "example")
    message.update(body=body, content_type="text/html")
    path = tmp_path / "inbox.json"
    path.write_text(json.dumps([message]))

    async def run():
        sample = state()
        await setup_email(str(path), str(tmp_path))(sample, None)
        init_subtask_store(sample.store)
        assert json.loads(await read_email()("example")) == message
        listing = json.loads(await read_email()())["messages"][0]
        assert listing["content_type"] == "text/html"
        assert listing["preview"] == body[:120]
        await send_email()("a@corp.example", "News", body, content_type="text/html")
        sent = sent_messages(sample.metadata["email_directory"])[0]
        assert sent["body"] == body
        assert sent["content_type"] == "text/html"

    asyncio.run(run())


@pytest.mark.parametrize("content_type", ["application/pdf", "text/html; charset=utf-8", "", None, []])
def test_invalid_content_type_rejected_for_inbox_and_send(tmp_path, content_type):
    message = dict.fromkeys(("id", "from", "to", "subject", "timestamp", "body"), "example")
    message["content_type"] = content_type
    path = tmp_path / "inbox.json"
    path.write_text(json.dumps([message]))
    with pytest.raises(ValueError, match="content_type"):
        load_inbox(str(path))

    async def run():
        sample = state()
        await setup_email(output_dir=str(tmp_path))(sample, None)
        init_subtask_store(sample.store)
        with pytest.raises(ToolError, match="content_type"):
            await send_email()("a@corp.example", "News", "body", content_type=content_type)
        assert sent_messages(sample.metadata["email_directory"]) == []

    asyncio.run(run())


def test_send_schema_exposes_optional_content_type_enum():
    parameters = ToolDef(send_email()).parameters
    assert parameters.properties["content_type"].enum == ["text/plain", "text/html"]
    assert "content_type" not in parameters.required
