from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .errors import AgentRuntimeError, ConcurrencyConflict, IntegrityViolation
from .events import AgentEvent, canonical_json, parse_canonical_payload, payload_digest
from .ids import validate_identifier
from .reducer import replay

SCHEMA_VERSION = 1


class SqliteAgentEventStore:
    def __init__(self, path: str | Path, *, durable: bool = True, busy_timeout_ms: int = 5_000) -> None:
        if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 60_000:
            raise ValueError("busy_timeout_ms must be between 1 and 60000")
        self.path = str(path)
        self.durable = durable
        self._connection = sqlite3.connect(self.path, timeout=busy_timeout_ms / 1000, isolation_level=None, check_same_thread=False)
        schema_version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if schema_version not in {0, SCHEMA_VERSION}:
            self._connection.close()
            raise IntegrityViolation("unknown_store_version", "unsupported SQLite event-store version")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        self._connection.execute(f"PRAGMA synchronous={'FULL' if durable else 'NORMAL'}")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_run_events(
              run_id TEXT NOT NULL, sequence INTEGER NOT NULL, event_id TEXT NOT NULL,
              event_version INTEGER NOT NULL, event_kind TEXT NOT NULL,
              occurred_at_utc TEXT NOT NULL, payload_json TEXT NOT NULL,
              payload_sha256 TEXT NOT NULL, action_id TEXT, invocation_id TEXT,
              budget_claim_id TEXT, PRIMARY KEY(run_id, sequence), UNIQUE(event_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_action_claim ON agent_run_events(action_id) WHERE action_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS uq_invocation_claim ON agent_run_events(invocation_id) WHERE event_kind='tool_proposed';
            CREATE UNIQUE INDEX IF NOT EXISTS uq_budget_claim ON agent_run_events(budget_claim_id) WHERE budget_claim_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS uq_idempotency_claim
              ON agent_run_events(json_extract(payload_json, '$.idempotency_key'))
              WHERE event_kind='authorization_recorded';
            """
        )
        if schema_version == 0:
            self._connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SqliteAgentEventStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def append(self, run_id: str, expected_version: int, events: Sequence[AgentEvent]) -> int:
        validate_identifier(run_id, field="run_id")
        if type(expected_version) is not int or expected_version < 0 or not events:
            raise ValueError("expected_version must be non-negative and events non-empty")
        rows = []
        for offset, event in enumerate(events, 1):
            payload_json = canonical_json(event.payload)
            rows.append((run_id, expected_version + offset, event.event_id, event.event_version, event.event_kind, event.occurred_at_utc, payload_json, payload_digest(payload_json), event.payload.get("action_id") if event.event_kind == "tool_proposed" else None, event.payload.get("invocation_id") if event.event_kind in {"tool_proposed", "tool_transition"} else None, event.payload.get("claim_id") if event.event_kind == "budget_consumed" else None))
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            actual = self._connection.execute("SELECT COALESCE(MAX(sequence),0) FROM agent_run_events WHERE run_id=?", (run_id,)).fetchone()[0]
            if actual != expected_version:
                raise ConcurrencyConflict("version_conflict", f"expected {expected_version}, found {actual}")
            existing = self.load(run_id)
            candidate = existing + tuple(
                replace(event, sequence=expected_version + offset)
                for offset, event in enumerate(events, 1)
            )
            replay(run_id, candidate)
            self._connection.executemany("INSERT INTO agent_run_events VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._connection.execute("COMMIT")
        except ConcurrencyConflict:
            self._connection.execute("ROLLBACK")
            raise
        except sqlite3.IntegrityError as error:
            self._connection.execute("ROLLBACK")
            raise IntegrityViolation("duplicate_claim", "event, action, invocation, or budget claim already exists") from error
        except AgentRuntimeError:
            self._connection.execute("ROLLBACK")
            raise
        return expected_version + len(events)

    def load(self, run_id: str) -> tuple[AgentEvent, ...]:
        validate_identifier(run_id, field="run_id")
        rows = self._connection.execute("SELECT sequence,event_id,event_version,event_kind,occurred_at_utc,payload_json,payload_sha256 FROM agent_run_events WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()
        result = []
        for expected, row in enumerate(rows, 1):
            sequence, event_id, version, kind, occurred, payload_json, digest = row
            if sequence != expected:
                raise IntegrityViolation("event_gap", "event sequence is not contiguous")
            if payload_digest(payload_json) != digest:
                raise IntegrityViolation("hash_mismatch", "event payload hash does not match")
            if version != 1:
                raise IntegrityViolation("unknown_event_version", "unsupported event schema version")
            try:
                payload = parse_canonical_payload(payload_json)
                event = AgentEvent(event_id, kind, payload, occurred, version, sequence)
            except AgentRuntimeError as error:
                raise IntegrityViolation("invalid_persisted_event", "persisted event is invalid") from error
            result.append(event)
        return tuple(result)
