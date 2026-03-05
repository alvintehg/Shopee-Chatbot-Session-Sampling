from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from .models import Conversation

def _in_date_range(ts: str, start: Optional[str], end: Optional[str], tz: str) -> bool:
    # Best-effort parsing. If parsing fails, keep it.
    if not start and not end:
        return True
    try:
        # accept ISO or 'YYYY-MM-DD HH:MM:SS'
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return True

    local = dt.astimezone(ZoneInfo(tz)) if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(tz))
    if start:
        s = datetime.fromisoformat(start).replace(tzinfo=ZoneInfo(tz))
        if local < s:
            return False
    if end:
        e = datetime.fromisoformat(end).replace(tzinfo=ZoneInfo(tz))
        if local > e:
            return False
    return True

def apply_filters(
    conversations: List[Conversation],
    start_date: Optional[str],
    end_date: Optional[str],
    include_intents: List[str],
    exclude_intents: List[str],
    include_node_ids: List[str],
    exclude_node_ids: List[str],
    min_messages_per_conversation: int,
    timezone: str,
) -> List[Conversation]:
    out: List[Conversation] = []
    inc_int = set([x.lower() for x in include_intents or []])
    exc_int = set([x.lower() for x in exclude_intents or []])
    inc_nodes = set(include_node_ids or [])
    exc_nodes = set(exclude_node_ids or [])

    for c in conversations:
        if c.message_count() < (min_messages_per_conversation or 0):
            continue

        # derive conversation-level intent/node from first non-null message
        if not c.intent:
            for m in c.messages:
                if m.intent:
                    c.intent = m.intent
                    break
        if not c.node_id:
            for m in c.messages:
                if m.node_id:
                    c.node_id = m.node_id
                    break
        if c.is_escalated is None:
            for m in c.messages:
                if m.is_escalated is not None:
                    c.is_escalated = m.is_escalated
                    break
        if c.rating is None:
            for m in c.messages:
                if m.rating is not None:
                    c.rating = m.rating
                    break
        if not c.issue_id:
            for m in c.messages:
                if m.issue_id:
                    c.issue_id = m.issue_id
                    break

        # date range check using first message timestamp
        ts0 = c.messages[0].timestamp if c.messages else ""
        if not _in_date_range(ts0, start_date, end_date, timezone):
            continue

        if inc_int and (not c.intent or c.intent.lower() not in inc_int):
            continue
        if exc_int and c.intent and c.intent.lower() in exc_int:
            continue
        if inc_nodes and (not c.node_id or c.node_id not in inc_nodes):
            continue
        if exc_nodes and c.node_id and c.node_id in exc_nodes:
            continue

        out.append(c)
    return out
