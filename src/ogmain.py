from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import yaml
from dotenv import load_dotenv

from .loader import (
    load_conversations_from_csv,
    load_conversations_from_json,
    load_conversations_from_jsonl,
    load_conversations_from_xlsx,
)
from .filters import apply_filters
from .issue_tagger import IssueTaggerConfig, annotate_issue_types
from .llm_classifier import LLMClassifierConfig, annotate_with_llm
from .sampler import sample_random, sample_stratified, sample_top_k_then_random
from .exporter import export_sample
from .summary import build_summary

def load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["file", "api", "web"], default="file")
    ap.add_argument("--input", type=str, default="data/raw", help="Folder containing exported transcripts")
    ap.add_argument("--out", type=str, default="data/out")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    load_dotenv()

    cfg = load_config(Path(args.config))

    tz = cfg["project"].get("timezone", "Asia/Kuala_Lumpur")

    # Load conversations
    if args.mode == "file":
        in_dir = Path(args.input)
        fmt = cfg["input"]["file_mode"].get("format", "jsonl").lower()
        if fmt == "jsonl":
            conversations = load_conversations_from_jsonl(in_dir)
        elif fmt == "json":
            conversations = load_conversations_from_json(in_dir)
        elif fmt == "csv":
            mapping = cfg["input"]["file_mode"].get("csv_mapping", {})
            conversations = load_conversations_from_csv(in_dir, mapping)
        elif fmt == "xlsx":
            mapping = cfg["input"]["file_mode"].get("xlsx_mapping", {})
            sheet_name = cfg["input"]["file_mode"].get("xlsx_sheet_name", "Data")
            conversations = load_conversations_from_xlsx(in_dir, mapping, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported file format: {fmt}")
    elif args.mode == "api":
        # API mode (stub)
        raise NotImplementedError(
            "API mode is a stub in this template. Wire src/shopee_admin_client.py to your internal endpoints."
        )
    else:
        from .web_collector import WebCollectorConfig, collect_conversations_from_web

        web_cfg = cfg.get("input", {}).get("web_mode", {})
        collector_cfg = WebCollectorConfig(
            login_url=web_cfg.get("login_url", ""),
            headless=bool(web_cfg.get("headless", False)),
            max_conversations=int(web_cfg.get("max_conversations", 200)),
            max_pages=int(web_cfg.get("max_pages", 3)),
            selectors=web_cfg.get("selectors", {}),
            filters=web_cfg.get("filters", []),
            wait_after_filter_seconds=float(web_cfg.get("wait_after_filter_seconds", 1.0)),
            post_login_url_contains=web_cfg.get("post_login_url_contains"),
            timezone=tz,
            filter_ready_selector=web_cfg.get("filter_ready_selector"),
            login_timeout_seconds=int(web_cfg.get("login_timeout_seconds", 300)),
            storage_state_path=web_cfg.get("storage_state_path", ".auth/storage_state.json"),
        )
        conversations = collect_conversations_from_web(
            collector_cfg,
            username=os.getenv("SHOPEE_ADMIN_USERNAME"),
            password=os.getenv("SHOPEE_ADMIN_PASSWORD"),
        )

    llm_cfg = cfg.get("llm_classification", {})
    llm_enabled = bool(llm_cfg.get("enabled", False))

    issue_cfg = cfg.get("issue_tagging", {})
    if issue_cfg.get("enabled", True) and not llm_enabled:
        conversations = annotate_issue_types(
            conversations,
            IssueTaggerConfig(
                taxonomy=issue_cfg.get("taxonomy", {}),
                default_issue_type=issue_cfg.get("default_issue_type", "other"),
                include_bot_messages=bool(issue_cfg.get("include_bot_messages", False)),
                overwrite_existing=bool(issue_cfg.get("overwrite_existing", False)),
                allowed_issue_types=issue_cfg.get("allowed_issue_types") or [],
                enforce_allowed_issue_types=bool(issue_cfg.get("enforce_allowed_issue_types", False)),
            ),
        )

    # Filters
    f = cfg.get("filters", {})
    conversations = apply_filters(
        conversations=conversations,
        start_date=f.get("start_date"),
        end_date=f.get("end_date"),
        include_intents=f.get("include_intents") or [],
        exclude_intents=f.get("exclude_intents") or [],
        include_node_ids=f.get("include_node_ids") or [],
        exclude_node_ids=f.get("exclude_node_ids") or [],
        min_messages_per_conversation=int(f.get("min_messages_per_conversation") or 0),
        timezone=tz,
    )

    # Sampling
    s = cfg.get("sampling", {})
    strategy = (s.get("strategy") or "random").lower()
    n = int(s.get("n_conversations") or 0)

    if strategy == "random":
        result = sample_random(conversations, n, seed=args.seed)
    elif strategy == "stratified":
        result = sample_stratified(
            conversations,
            n,
            stratify_by=s.get("stratify_by") or ["intent"],
            max_per_stratum=s.get("max_per_stratum"),
            seed=args.seed,
        )
    elif strategy == "top_k":
        result = sample_top_k_then_random(
            conversations,
            n,
            top_k_field=s.get("top_k_field") or "is_escalated",
            top_k=int(s.get("top_k") or 100),
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if llm_enabled:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("llm_classification.enabled=true requires OPENAI_API_KEY in .env")

        result.conversations = annotate_with_llm(
            result.conversations,
            LLMClassifierConfig(
                model=llm_cfg.get("model", "gpt-4.1-mini"),
                max_messages=int(llm_cfg.get("max_messages", 30)),
                max_chars=int(llm_cfg.get("max_chars", 8000)),
                temperature=float(llm_cfg.get("temperature", 0.0)),
                request_timeout_s=int(llm_cfg.get("request_timeout_s", 45)),
                include_bot_messages=bool(llm_cfg.get("include_bot_messages", True)),
                default_issue=llm_cfg.get("default_issue", "Unclear"),
                default_issue_type=llm_cfg.get("default_issue_type", issue_cfg.get("default_issue_type", "Unclear")),
                allowed_issues=(cfg.get("review_labeling", {}).get("issue", {}).get("allowed") or []),
                allowed_issue_types=(issue_cfg.get("allowed_issue_types") or []),
            ),
            api_key=api_key,
            api_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    out_dir = Path(args.out)
    filenames = cfg["output"]
    export_info = export_sample(
        result.conversations,
        out_dir,
        filenames,
        timezone=tz,
        review_labeling=cfg.get("review_labeling", {}),
    )

    gs_cfg = cfg.get("google_sheets", {})
    if gs_cfg.get("enabled", False):
        from .google_sheets_uploader import upload_review_to_google_sheets

        review_csv = export_info.get("review_csv")
        if not review_csv:
            raise ValueError("google_sheets.enabled=true requires output.review_csv in config.")

        service_account_path = gs_cfg.get("service_account_json_path") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH")
        spreadsheet_id = gs_cfg.get("spreadsheet_id")
        worksheet_gid = gs_cfg.get("worksheet_gid")
        write_mode = (gs_cfg.get("write_mode") or "append").lower()
        create_new_tab = bool(gs_cfg.get("create_new_tab", False))
        new_tab_title = gs_cfg.get("new_tab_title")

        if not service_account_path:
            raise ValueError("Missing service account json path. Set google_sheets.service_account_json_path or GOOGLE_SERVICE_ACCOUNT_JSON_PATH.")
        if not spreadsheet_id:
            raise ValueError("Missing google_sheets.spreadsheet_id in config.")

        gsheet_info = upload_review_to_google_sheets(
            review_csv_path=Path(review_csv),
            service_account_json_path=Path(service_account_path),
            spreadsheet_id=str(spreadsheet_id),
            worksheet_gid=int(worksheet_gid) if worksheet_gid is not None else None,
            write_mode=write_mode,
            create_new_tab=create_new_tab,
            new_tab_title=new_tab_title,
            timezone=tz,
        )
        export_info["google_sheets"] = gsheet_info

    # Summary
    summ = build_summary(result.conversations)
    summ.update({"sampling": result.summary, "export": export_info})
    (out_dir / filenames["summary_json"]).write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done")
    print(json.dumps(export_info, indent=2))

if __name__ == "__main__":
    main()
