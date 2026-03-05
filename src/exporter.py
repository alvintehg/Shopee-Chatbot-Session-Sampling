"""
exporter.py  (PRODUCTION REWRITE)
──────────────────────────────────
Key fixes:
  ✅  Batched pandas export (vs row-by-row CSV writes)
  ✅  Consistent column order regardless of input shape
  ✅  UTF-8-BOM for Excel compatibility
  ✅  Separate review CSV with labeling columns pre-populated
  ✅  Export summary with row counts
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .models import Conversation

logger = logging.getLogger(__name__)

# Canonical column order for the review CSV
REVIEW_COLUMNS = [
    "id",
    "intent",
    "issue",
    "issue_type",
    "issue_confidence",
    "is_escalated",
    "message_count",
    "started_at",
    "full_text",
    # reviewer labeling columns (blank by default)
    "reviewer_issue",
    "reviewer_issue_type",
    "reviewer_notes",
    "reviewed_by",
    "reviewed_at",
]


def export_to_csv(
    conversations: list[Conversation],
    output_path: Path,
    review_mode: bool = False,
    chunksize: int = 500,
) -> int:
    """
    Export conversations to CSV.
    Returns number of rows written.
    """
    if not conversations:
        logger.warning("No conversations to export.")
        return 0

    rows = [c.to_csv_row() for c in conversations]
    df = pd.DataFrame(rows)

    if review_mode:
        # Add blank reviewer columns
        for col in REVIEW_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        # Enforce column order (only columns that exist)
        ordered = [c for c in REVIEW_COLUMNS if c in df.columns]
        df = df[ordered]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write in chunks to avoid memory spikes on large datasets
    first_chunk = True
    total_rows = 0
    for i in range(0, len(df), chunksize):
        chunk = df.iloc[i : i + chunksize]
        chunk.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",       # BOM for Excel
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        total_rows += len(chunk)

    logger.info(f"Exported {total_rows} rows → {output_path}")
    return total_rows


def export_sample(
    conversations: list[Conversation],
    out_dir: Path,
    filenames: dict,
    timezone: str = "Asia/Kuala_Lumpur",
    review_labeling: dict | None = None,
) -> dict:
    """
    Export full sample CSV + review CSV.
    Returns export info dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}

    # Full sample
    if "sample_csv" in filenames:
        path = out_dir / filenames["sample_csv"]
        n = export_to_csv(conversations, path, review_mode=False)
        info["sample_csv"] = str(path)
        info["sample_csv_rows"] = n

    # Review CSV (with blank labeling columns)
    if "review_csv" in filenames:
        path = out_dir / filenames["review_csv"]
        n = export_to_csv(conversations, path, review_mode=True)
        info["review_csv"] = str(path)
        info["review_csv_rows"] = n

    return info
