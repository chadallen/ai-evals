"""Email tools with host-only inbox data and per-sample sent artifacts."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from inspect_ai.solver import Solver, solver
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import store

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "all-logs" / "email"


def load_inbox(path: str | None) -> list[dict]:
    """Reject malformed fixtures before starting a sample."""
    messages = json.loads(Path(path).read_text()) if path else []
    if not isinstance(messages, list):
        raise ValueError("Inbox must be a JSON array of messages")
    ids = set()
    for message in messages:
        if not isinstance(message, dict) or any(
            not isinstance(message.get(key), str)
            for key in ("id", "from", "to", "subject", "timestamp", "body")
        ):
            raise ValueError(
                "Each inbox message needs string id, from, to, subject, timestamp, body"
            )
        if not message["id"] or message["id"] in ids:
            raise ValueError("Inbox message IDs must be nonempty and unique")
        ids.add(message["id"])
    return messages


@solver
def setup_email(inbox_file: str | None = None, output_dir: str | None = None) -> Solver:
    """Initialize an isolated mailbox for every sample, epoch, and retry."""
    inbox = load_inbox(inbox_file)
    root = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR

    async def solve(state, generate):
        # UUIDs avoid path injection from sample IDs and collisions across concurrent runs.
        directory = root / uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "sent").mkdir()
        (directory / "sample.json").write_text(
            json.dumps({"sample_id": state.sample_id, "epoch": state.epoch}, indent=2) + "\n"
        )
        state.metadata["email_directory"] = str(directory)
        state.store.set("email_directory", str(directory))
        state.store.set("email_inbox", inbox)
        return state

    return solve


@tool
def read_email() -> Tool:
    """Read email from your inbox."""

    async def execute(message_id: str | None = None) -> str:
        """List inbox messages, or read one message in full.

        Args:
            message_id: Message ID to read. Omit to list sender, subject, time, and preview.
        """
        inbox = store().get("email_inbox")
        if inbox is None:
            raise RuntimeError("Email tools require setup_email before use")
        if message_id is not None:
            message = next((m for m in inbox if m["id"] == message_id), None)
            if message is None:
                raise ToolError("Message not found.")
            return json.dumps(message, ensure_ascii=False)
        return json.dumps(
            {
                "messages": [
                    {
                        "id": m["id"],
                        "from": m["from"],
                        "subject": m["subject"],
                        "timestamp": m["timestamp"],
                        "preview": m["body"][:120],
                    }
                    for m in inbox
                ],
                "total": len(inbox),
            },
            ensure_ascii=False,
        )

    return execute


@tool
def send_email() -> Tool:
    """Send an email."""

    async def execute(to: str, subject: str, body: str) -> str:
        """Send an email and return its message ID and status.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Plain-text email body.
        """
        if not to.strip() or "@" not in to or any(c in to for c in "\r\n"):
            raise ToolError("Provide a recipient email address.")
        if any(c in subject for c in "\r\n"):
            raise ToolError("Subject must be a single line.")
        directory = store().get("email_directory")
        if directory is None:
            raise RuntimeError("Email tools require setup_email before use")
        message_id = uuid4().hex
        message = {
            "id": message_id,
            "to": to.strip(),
            "subject": subject,
            "body": body,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        sent = Path(directory) / "sent"
        pending = sent / f"{message_id}.tmp"
        try:
            # Only completed JSON files count as sends, including after an interrupted write.
            with pending.open("x") as output:
                output.write(json.dumps(message, ensure_ascii=False, indent=2) + "\n")
            pending.replace(sent / f"{message_id}.json")
        except OSError as exc:
            pending.unlink(missing_ok=True)
            raise ToolError("Unable to send email. Please try again.") from exc
        return json.dumps({"id": message_id, "status": "sent"})

    return execute


def sent_messages(directory: str) -> list[dict]:
    """Read persisted sends; missing artifacts are an error, not a zero score."""
    sent = Path(directory) / "sent"
    if not sent.is_dir():
        raise FileNotFoundError(f"Sent-mail artifacts are unavailable: {sent}")
    return sorted(
        [json.loads(path.read_text()) for path in sent.glob("*.json")],
        key=lambda message: (message["timestamp"], message["id"]),
    )
