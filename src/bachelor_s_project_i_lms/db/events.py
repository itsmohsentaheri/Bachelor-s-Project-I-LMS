# src/bachelor_s_project_i_lms/db/events.py
"""Append-only event logging."""
import os
from typing import Optional
from uuid import UUID

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

load_dotenv()


def log_event(
    type: str,
    payload: dict,
    learner_id: Optional[UUID] = None,
    graph_version: Optional[int] = None,
    model_version: Optional[str] = None,
    prompt_version: Optional[str] = None,
) -> int:
    """Insert one row into `events` and return its event_id."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment variables.")

    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            """
            INSERT INTO events
                (type, payload, learner_id, graph_version, model_version, prompt_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING event_id
            """,
            (type, Jsonb(payload), learner_id, graph_version,
             model_version, prompt_version),
        ).fetchone()
        return row[0]
