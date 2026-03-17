from __future__ import annotations

import os
from typing import Any

import requests


class _BaseJsonClient:
    def __init__(self, base_url: str, *, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get_json(self, endpoint: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params or {}, headers=headers or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


class SportmonksClient(_BaseJsonClient):
    def __init__(self, api_token: str | None = None, *, base_url: str = "https://api.sportmonks.com/v3/football", timeout: int = 30) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self.api_token = api_token or os.getenv("SPORTMONKS_API_TOKEN", "")

    def get_json(self, endpoint: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        query = dict(params or {})
        if self.api_token:
            query.setdefault("api_token", self.api_token)
        return super().get_json(endpoint, params=query, headers=headers)


class ApiFootballClient(_BaseJsonClient):
    def __init__(self, api_key: str | None = None, *, base_url: str = "https://v3.football.api-sports.io", timeout: int = 30) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self.api_key = api_key or os.getenv("APIFOOTBALL_API_KEY", "")

    def get_json(self, endpoint: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        hdrs = dict(headers or {})
        if self.api_key:
            hdrs.setdefault("x-apisports-key", self.api_key)
        return super().get_json(endpoint, params=params, headers=hdrs)
