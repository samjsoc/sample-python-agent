"""Persistent task assistant powered by Pydantic AI and local Ollama.

Run:
    uv add pydantic-ai
    uv run task_agent.py

Environment:
    OLLAMA_MODEL=qwen3.6:35b
    OLLAMA_BASE_URL=http://localhost:11434/v1
    TASK_TIMEZONE=America/Los_Angeles
    TASK_DB=tasks.db
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import UsageLimits


class Task(BaseModel):
    id: str
    description: str
    due_at: datetime | None = None
    created_at: datetime
    completed_at: datetime | None = None
    completed: bool = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskService:
    """Database operations stay independent of the LLM/framework."""
        
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    due_at TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)

    @staticmethod
    def _to_task(row: sqlite3.Row) -> Task:
        completed_at = (
            datetime.fromisoformat(row["completed_at"])
            if row["completed_at"] else None
        )
        return Task(
            id=row["id"],
            description=row["description"],
            due_at=(datetime.fromisoformat(row["due_at"]) if row["due_at"] else None),
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=completed_at,
            completed=completed_at is not None,
        )

    def add(self, description: str, due_at: datetime | None = None) -> Task:
        description = description.strip()
        if not description:
            raise ValueError("Task description cannot be blank")
        if due_at is not None and due_at.tzinfo is None:
            raise ValueError("Deadline must include a timezone offset")
        task = Task(
            id=str(uuid4()),
            description=description,
            due_at=due_at.astimezone(timezone.utc) if due_at else None,
            created_at=utc_now(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks (id, description, due_at, created_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task.id, task.description,
                 task.due_at.isoformat() if task.due_at else None,
                 task.created_at.isoformat(), None),
            )
        return task

    def list(self, status: Literal["all", "open", "completed"] = "open") -> list[Task]:
        where = {
            "all": "",
            "open": "WHERE completed_at IS NULL",
            "completed": "WHERE completed_at IS NOT NULL",
        }[status]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks {where} ORDER BY due_at IS NULL, due_at, created_at"
            ).fetchall()
        return [self._to_task(row) for row in rows]

    def complete(self, task_id: str) -> Task:
        with self._connect() as conn:
            existing = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if existing is None:
                raise ValueError("No task exists with that ID")
            if existing["completed_at"] is None:
                conn.execute(
                    "UPDATE tasks SET completed_at = ? WHERE id = ?",
                    (utc_now().isoformat(), task_id),
                )
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._to_task(row)


TIMEZONE_NAME = os.getenv("TASK_TIMEZONE", "America/Los_Angeles")
try:
    LOCAL_TZ = ZoneInfo(TIMEZONE_NAME)
except ZoneInfoNotFoundError as exc:
    raise SystemExit(f"Unknown TASK_TIMEZONE: {TIMEZONE_NAME}") from exc

service = TaskService(os.getenv("TASK_DB", "tasks.db"))
model = OllamaModel(
    os.getenv("OLLAMA_MODEL", "qwen3.6:35b"),
    provider=OllamaProvider(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ),
)
agent = Agent(
    model,
    instructions=(
        "You are a helpful task-management assistant. "
        "Use tools to read or modify tasks; never invent task IDs or claim a task "
        "was saved unless a tool confirms it. "
        f"The user's timezone is {TIMEZONE_NAME}. "
        "Call get_current_time before resolving relative deadlines such as "
        "'tomorrow' or 'in an hour'. "
        "Pass due_at as an RFC 3339 timestamp with an explicit UTC offset "
        "(e.g. 2026-10-15T09:00:00-07:00). "
        "Ask a clarification question if the time is ambiguous. "
        "When marking a task complete, first list tasks to find its exact ID. "
        "Never call a tool to execute arbitrary shell commands."
    ),
)


@agent.tool_plain
def get_current_time() -> str:
    """Get the current local time (including UTC offset) for relative deadlines."""
    return utc_now().astimezone(LOCAL_TZ).isoformat()


@agent.tool_plain
def add_task(description: str, due_at: str | None = None) -> Task | str:
    """Create a task; due_at is optional RFC 3339 with an explicit timezone offset."""
    try:
        parsed = None
        if due_at is not None:
            parsed = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
            if parsed.utcoffset() is None:
                return "Invalid due_at: include a timezone offset, e.g. -07:00 or Z."
        return service.add(description, parsed)
    except ValueError as exc:
        return f"Could not create task: {exc}"


@agent.tool_plain
def list_tasks(status: Literal["all", "open", "completed"] = "open") -> list[Task]:
    """List tasks and IDs. Status may be open, completed, or all."""
    return service.list(status)


@agent.tool_plain
def complete_task(task_id: str) -> Task | str:
    """Mark a task completed using its exact ID from list_tasks."""
    try:
        return service.complete(task_id)
    except ValueError as exc:
        return f"Could not complete task: {exc}"


def main() -> None:
    print(f"Task assistant | model={os.getenv('OLLAMA_MODEL', 'qwen3.6:35b')} "
          f"| timezone={TIMEZONE_NAME} | db={service.db_path}")
    print("Type 'exit' to quit.\n")
    history = None
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        try:
            result = agent.run_sync(
                prompt,
                message_history=history,
                usage_limits=UsageLimits(request_limit=12),
            )
            history = result.all_messages()
            print(f"Agent: {result.output}\n")
        except Exception as exc:
            print(f"Agent error: {exc}\n")


if __name__ == "__main__":
    main()
