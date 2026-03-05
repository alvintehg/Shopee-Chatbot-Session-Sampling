from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests

@dataclass
class AdminClientConfig:
    base_url: str
    bearer_token: Optional[str] = None
    cookie: Optional[str] = None
    timeout_s: int = 30

class ShopeeBotAdminClient:
    """API-mode client (STUB).

    Replace endpoints/params with your internal Bot Admin APIs.
    Keep the return schema compatible with src/models.py (Conversation).
    """

    def __init__(self, cfg: AdminClientConfig):
        self.cfg = cfg
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "transcript-sampler/1.0"})
        if cfg.bearer_token:
            self.s.headers.update({"Authorization": f"Bearer {cfg.bearer_token}"})
        if cfg.cookie:
            self.s.headers.update({"Cookie": cfg.cookie})

    def healthcheck(self) -> bool:
        # TODO: adjust endpoint
        url = f"{self.cfg.base_url.rstrip('/')}/health"
        try:
            r = self.s.get(url, timeout=self.cfg.timeout_s)
            return r.status_code < 400
        except Exception:
            return False

    def list_transcripts(self, start_date: str, end_date: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Return a list of transcript 'headers' (ids + metadata)."""
        # TODO: implement with your internal endpoint
        raise NotImplementedError("Wire this to your Bot Admin list/search endpoint.")

    def get_transcript(self, conversation_id: str) -> Dict[str, Any]:
        """Return raw transcript payload for one conversation."""
        # TODO: implement with your internal endpoint
        raise NotImplementedError("Wire this to your Bot Admin transcript detail endpoint.")
