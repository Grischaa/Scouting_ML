from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from scouting_ml.tests.test_market_value_api import _write_clean_profile_dataset, _write_profile_artifacts

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright


def _find_open_port() -> int:
    """Return an available localhost TCP port for the temporary test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_backend(base_url: str, process: subprocess.Popen[str], timeout_s: float = 45.0) -> None:
    """Wait until the FastAPI backend answers its health endpoint or raise with captured logs."""
    deadline = time.time() + timeout_s
    last_error: str | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise RuntimeError(f"Temporary backend exited early.\n{output}")
        try:
            response = httpx.get(f"{base_url}/market-value/health", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"health returned {response.status_code}: {response.text}"
        except Exception as exc:  # pragma: no cover - network/process race during startup
            last_error = str(exc)
        time.sleep(0.5)
    output = process.stdout.read() if process.stdout is not None else ""
    raise RuntimeError(f"Temporary backend did not become ready: {last_error}\n{output}")


@pytest.fixture()
def player_detail_backend(tmp_path: Path) -> Iterator[str]:
    """Start a temporary backend with profile, similarity, trajectory, and proxy data enabled."""
    test_path, val_path, metrics_path = _write_profile_artifacts(tmp_path)
    clean_path = _write_clean_profile_dataset(tmp_path, test_path)

    port = _find_open_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": "src",
            "SCOUTING_TEST_PREDICTIONS_PATH": str(test_path),
            "SCOUTING_VAL_PREDICTIONS_PATH": str(val_path),
            "SCOUTING_METRICS_PATH": str(metrics_path),
            "SCOUTING_CLEAN_DATASET_PATH": str(clean_path),
            "SCOUTING_WATCHLIST_PATH": str(tmp_path / "watchlist.jsonl"),
            "SCOUTING_DECISIONS_PATH": str(tmp_path / "scout_decisions.jsonl"),
            "SCOUTING_API_CORS_ORIGINS": base_url,
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "scouting_ml.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_backend(base_url, process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - cleanup safeguard
            process.kill()
            process.wait(timeout=5)


def _connect_and_wait_for_board(page, base_url: str) -> None:
    """Load the mounted frontend, point it at the temp backend, and wait for board rows."""
    page.goto(f"{base_url}/app/index.html", wait_until="networkidle")
    page.locator("#api-base").fill(base_url)
    page.locator("#connect-btn").click()
    page.locator("#results-body tr[data-index]").first.wait_for(timeout=30000)


def _open_player_from_board(page, player_name: str) -> None:
    """Filter the board to one player, click the row, and wait for the detail rail to hydrate."""
    page.locator("#search-input").fill(player_name)
    page.locator("#refresh-btn").click()
    row = page.locator("#results-body tr[data-index]").filter(has_text=player_name).first
    row.wait_for(timeout=30000)
    row.click()
    page.locator("#detail-content").wait_for(timeout=30000)
    page.locator("#detail-name").filter(has_text=player_name).wait_for(timeout=30000)


@pytest.mark.e2e
def test_player_detail_flow_regression(player_detail_backend: str) -> None:
    """Exercise the player-detail flow including decision logging, similar, trajectory, memo export, and proxy estimates."""
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except Exception as exc:  # pragma: no cover - depends on local browser install
                pytest.skip(f"Playwright browser unavailable: {exc}")
            with browser:
                page = browser.new_page()
                request_log: list[str] = []
                page.on("request", lambda request: request_log.append(request.url))
                _connect_and_wait_for_board(page, player_detail_backend)
                assert any("/market-value/ui-bootstrap" in url for url in request_log)
                assert not any("/market-value/system-fit/templates" in url for url in request_log)

                before_open = len(request_log)
                _open_player_from_board(page, "Profile Target")
                initial_open_requests = request_log[before_open:]
                assert any("/market-value/player/profile_fw_target/profile" in url for url in initial_open_requests)
                assert not any("/market-value/player/profile_fw_target/similar" in url for url in initial_open_requests)
                assert not any("/market-value/player/profile_fw_target/trajectory" in url for url in initial_open_requests)

                page.locator('[data-detail-tab="tactical"]').click()
                page.locator("#detail-similar [data-similar-player-id]").first.wait_for(timeout=30000)
                assert page.locator("#detail-similar-summary").text_content()
                assert page.locator("#detail-proxy-section").is_hidden()

                before_trajectory = len(request_log)
                page.locator('[data-detail-tab="trajectory"]').click()
                page.locator("#detail-trajectory-chart svg").wait_for(timeout=30000)
                page.locator("#detail-trajectory-table-body tr").first.wait_for(timeout=30000)
                trajectory_requests = [
                    url for url in request_log[before_trajectory:] if "/market-value/player/profile_fw_target/trajectory" in url
                ]
                assert len(trajectory_requests) == 1

                page.locator('[data-detail-tab="overview"]').click()
                page.locator('[data-detail-tab="trajectory"]').click()
                page.locator("#detail-trajectory-chart svg").wait_for(timeout=30000)
                assert len([url for url in request_log if "/market-value/player/profile_fw_target/trajectory" in url]) == 1

                page.locator('[data-detail-tab="risk"]').click()
                page.locator('[data-scout-action="shortlist"]').click()
                page.locator('[data-decision-reason="system_fit"]').click()
                page.locator("#detail-decision-save-btn").click()
                page.locator("#detail-latest-decision").wait_for(timeout=30000)
                assert "Shortlist" in (page.locator("#detail-latest-decision-pill").text_content() or "")
                page.locator("#watchlist-body tr").filter(has_text="Profile Target").first.wait_for(timeout=30000)

                with page.expect_download(timeout=30000) as download_info:
                    page.locator("#detail-export-pdf").click()
                download = download_info.value
                assert download.suggested_filename.endswith(".pdf")

                page.locator('[data-detail-tab="tactical"]').click()
                initial_name = (page.locator("#detail-name").text_content() or "").strip()
                similar_card = page.locator("#detail-similar [data-similar-player-id]").first
                similar_card.click()
                page.wait_for_function(
                    """(previousName) => {
                        const el = document.querySelector("#detail-name");
                        return !!el && !!el.textContent && el.textContent.trim() !== previousName;
                    }""",
                    arg=initial_name,
                    timeout=30000,
                )

                _open_player_from_board(page, "Profile Sparse")
                page.locator('[data-detail-tab="tactical"]').click()
                page.locator("#detail-proxy-section").wait_for(timeout=30000)
                assert page.locator("#detail-proxy-section").is_visible()
                assert "Derived from comparable-player neighbors" in (
                    page.locator("#detail-proxy-summary").text_content() or ""
                )
                page.locator("#detail-proxy-list li").first.wait_for(timeout=30000)

                page.locator("#watchlist-body .watchlist-open-decision").filter(has_text="Update decision").first.click()
                page.locator('[data-scout-action="pass"]').click()
                page.locator('[data-decision-reason="league_risk"]').click()
                page.locator("#detail-decision-save-btn").click()
                page.locator("#detail-latest-decision-pill").filter(has_text="Pass").wait_for(timeout=30000)
                assert page.locator("#watchlist-body tr").filter(has_text="Profile Target").count() == 1
    except pytest.skip.Exception:
        raise
