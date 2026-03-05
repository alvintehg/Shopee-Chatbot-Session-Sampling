from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import gspread
from google.oauth2.service_account import Credentials
from zoneinfo import ZoneInfo

SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]


def _open_worksheet_by_gid(spreadsheet: gspread.Spreadsheet, gid: int):
    for ws in spreadsheet.worksheets():
        if ws.id == gid:
            return ws
    raise ValueError(f"Worksheet with gid={gid} not found in spreadsheet")


def _normalize_row(row: Dict[str, object], headers: List[str]) -> List[str]:
    out: List[str] = []
    for h in headers:
        v = row.get(h, "")
        if v is None:
            out.append("")
        else:
            out.append(str(v))
    return out


def _build_unique_title(spreadsheet: gspread.Spreadsheet, base_title: str) -> str:
    existing = set([ws.title for ws in spreadsheet.worksheets()])
    if base_title not in existing:
        return base_title
    i = 2
    while True:
        candidate = f"{base_title} - {i}"
        if candidate not in existing:
            return candidate
        i += 1


def upload_review_to_google_sheets(
    review_csv_path: Path,
    service_account_json_path: Path,
    spreadsheet_id: str,
    worksheet_gid: Optional[int],
    write_mode: str = "append",
    create_new_tab: bool = False,
    new_tab_title: Optional[str] = None,
    timezone: str = "Asia/Kuala_Lumpur",
) -> Dict[str, object]:
    if not review_csv_path.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_csv_path}")
    if not service_account_json_path.exists():
        raise FileNotFoundError(f"Service account key not found: {service_account_json_path}")

    import pandas as pd

    df = pd.read_csv(review_csv_path)
    headers = list(df.columns)
    rows = df.fillna("").to_dict(orient="records")

    creds = Credentials.from_service_account_file(str(service_account_json_path), scopes=SHEETS_SCOPES)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key(spreadsheet_id)

    if create_new_tab:
        base_title = (new_tab_title or "").strip()
        if not base_title:
            base_title = datetime.now(ZoneInfo(timezone)).strftime("Review_%Y-%m-%d_%H%M")
        tab_title = _build_unique_title(spreadsheet, base_title)
        ws = spreadsheet.add_worksheet(title=tab_title, rows=max(1000, len(rows) + 20), cols=max(20, len(headers) + 5))
        write_mode = "replace"
    elif worksheet_gid is not None:
        ws = _open_worksheet_by_gid(spreadsheet, worksheet_gid)
    else:
        ws = spreadsheet.sheet1

    if write_mode not in {"append", "replace"}:
        raise ValueError("write_mode must be 'append' or 'replace'")

    values = [_normalize_row(r, headers) for r in rows]

    if write_mode == "replace":
        ws.clear()
        ws.append_row(headers, value_input_option="USER_ENTERED")
        if values:
            ws.append_rows(values, value_input_option="USER_ENTERED")
    else:
        existing_header = ws.row_values(1)
        if not existing_header:
            ws.append_row(headers, value_input_option="USER_ENTERED")
        elif existing_header != headers:
            raise ValueError(
                "Worksheet header does not match review columns. "
                "Use write_mode=replace or align sheet headers first."
            )
        if values:
            ws.append_rows(values, value_input_option="USER_ENTERED")

    return {
        "spreadsheet_id": spreadsheet_id,
        "worksheet_title": ws.title,
        "worksheet_gid": ws.id,
        "write_mode": write_mode,
        "rows_uploaded": len(values),
        "headers": headers,
    }
