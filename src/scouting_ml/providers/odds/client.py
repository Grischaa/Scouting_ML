from __future__ import annotations

import os
from typing import Any

import requests


class OddsApiClient:
    def __init__(self, api_key: str | None = None, *, base_url: str = "https://api.the-odds-api.com/v4", timeout: int = 30) -> None:
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get_json(self, endpoint: str, *, params: dict[str, Any] | None = None) -> Any:
        query = dict(params or {})
        if self.api_key:
            query.setdefault("apiKey", self.api_key)
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=query, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
