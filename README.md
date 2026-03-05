# Shopee Bot Admin Transcript Sampler (VS Code Template)

This project lets you:
1) fetch transcripts from Bot Admin (from exports, API, or web UI automation),
2) normalize to one schema,
3) filter and sample conversations,
4) auto-mark `issue_type`,
5) export CSV/JSONL for review.

## Quick start

### 1) Create and activate venv
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 3) Copy environment variables
```bash
copy .env.example .env
```

### 4) Configure sampling and selectors
Edit `config.yaml`:
- `input.file_mode` for file import, or
- `input.web_mode` for Admin website automation,
- `issue_tagging.taxonomy` for keyword categories, or
- `llm_classification` for LLM-based Issue/Issue Type classification.

## Run modes

### File mode (works immediately)
Put exported transcripts in `data/raw/` then run:
```bash
python -m src.main --mode file --input data/raw --out data/out
```

### Web mode (Admin login + filter + transcript read)
1. Fill `input.web_mode.login_url` and `input.web_mode.selectors` in `config.yaml`.
2. Optional: set `.env` with `SHOPEE_ADMIN_USERNAME` and `SHOPEE_ADMIN_PASSWORD`.
3. Run:
```bash
python -m src.main --mode web --out data/out
```

`web` mode behavior:
- opens Admin page with Playwright,
- logs in (if username/password selectors and env vars are configured),
- saves/reuses session state from `input.web_mode.storage_state_path` (default `.auth/storage_state.json`),
- applies filter actions from `input.web_mode.filters`,
- reads transcript pages and extracts messages,
- tags each conversation with `issue_type` (keyword or LLM mode),
- applies normal filters/sampling/export pipeline.

## LLM Classification (Optional)
Enable `llm_classification.enabled: true` in `config.yaml`, then set in `.env`:
- `OPENAI_API_KEY=...`
- optional `OPENAI_BASE_URL=...` (defaults to `https://api.openai.com/v1`)

When enabled, sampled conversations are classified by LLM for both:
- `Issue` (review column)
- `Issue Type` (review + conversation field)

Values are constrained to your allowed dropdown labels in config.

### API mode
Still a stub. Implement in `src/shopee_admin_client.py`.

## Output
- `data/out/sample_conversations.jsonl` (conversation-level)
- `data/out/sample_messages.csv` (message-level)
- `data/out/sample_review.csv` (QA review columns: Session Date, Session ID, Intent, Node Point ID, Deflected?, CSAT, Issue, Correct Intent, Issue Type, Description)
- `data/out/summary.json` (counts + distributions + issue types)

## Push To Google Sheets
You can auto-upload `sample_review.csv` to a Google Sheet tab.

1. Create a Google Cloud service account and download the JSON key.
2. Share the target Google Sheet with the service account email (Editor).
3. Set either:
   - `google_sheets.service_account_json_path` in `config.yaml`, or
   - `GOOGLE_SERVICE_ACCOUNT_JSON_PATH` in `.env`.
4. In `config.yaml`, set:
   - `google_sheets.enabled: true`
   - `google_sheets.spreadsheet_id`
   - `google_sheets.worksheet_gid`
   - `google_sheets.write_mode: append` or `replace`
   - optional: `google_sheets.create_new_tab: true` to create a fresh worksheet tab each run

When enabled, the pipeline uploads the review rows after export.

## Important notes
- Admin DOM differs by team. Update selectors in `config.yaml` to match your actual website.
- If your Admin uses SSO/MFA/captcha, keep `headless: false` and complete auth manually, then continue collection.
- Do not commit `.env` or raw transcripts.

## Main files
- `src/web_collector.py`: web UI automation and transcript extraction
- `src/issue_tagger.py`: issue type classification by keywords
- `src/main.py`: mode selection, filtering, sampling, export
