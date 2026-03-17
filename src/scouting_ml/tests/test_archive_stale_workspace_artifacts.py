from __future__ import annotations

import json
from pathlib import Path

from scouting_ml.scripts.archive_stale_workspace_artifacts import archive_stale_workspace_artifacts


def test_archive_stale_workspace_artifacts_moves_only_superseded_files(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    data = repo / "data"
    processed = data / "processed"
    incoming = processed / "_incoming_future"
    legacy = processed / "_with_sofa"
    combined = processed / "Clubs combined"
    model = data / "model"
    reports = model / "reports"
    for path in [incoming, legacy, combined, model, reports]:
        path.mkdir(parents=True, exist_ok=True)

    root_live = processed / "scottish_premiership_2025-26_with_sofa.csv"
    root_live.write_text("player_id\np1\n", encoding="utf-8")
    (incoming / "scottish_premiership_2025-26_with_sofa.csv").write_text("player_id\np1\n", encoding="utf-8")
    (legacy / "scottish_premiership_2025-26_with_sofa.csv").write_text("player_id\nold\n", encoding="utf-8")
    (legacy / "estonian_meistriliiga_2025_with_sofa.csv").write_text("player_id\nest\n", encoding="utf-8")
    (model / "_tmp_smoke.csv").write_text("x\n", encoding="utf-8")

    manifest = processed / "organization_manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "basename,league_slug,season,country,source_path,combined_path,country_path,season_path,duplicate_count,duplicate_collision",
                "scottish_premiership_2025-26_with_sofa.csv,scottish_premiership,2025-26,scotland,data/processed/scottish_premiership_2025-26_with_sofa.csv,data/processed/Clubs combined/scottish_premiership_2025-26_with_sofa.csv,x,x,2,False",
                "estonian_meistriliiga_2025_with_sofa.csv,estonian_meistriliiga,2025,estonia,data/processed/_with_sofa/estonian_meistriliiga_2025_with_sofa.csv,data/processed/Clubs combined/estonian_meistriliiga_2025_with_sofa.csv,x,x,1,False",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    summary = archive_stale_workspace_artifacts(execute=True)

    assert summary["candidate_count"] == 3
    assert not (incoming / "scottish_premiership_2025-26_with_sofa.csv").exists()
    assert not (legacy / "scottish_premiership_2025-26_with_sofa.csv").exists()
    assert not (model / "_tmp_smoke.csv").exists()

    assert (legacy / "estonian_meistriliiga_2025_with_sofa.csv").exists()

    archive_root = Path(str(summary["archive_root"]))
    assert (archive_root / "data/processed/_incoming_future/scottish_premiership_2025-26_with_sofa.csv").exists()
    assert (archive_root / "data/processed/_with_sofa/scottish_premiership_2025-26_with_sofa.csv").exists()
    assert (archive_root / "data/model/_tmp_smoke.csv").exists()

    manifest_path = Path(str(summary["archive_manifest"]))
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["moved_count"] == 3


def test_archive_stale_workspace_artifacts_moves_old_workflow_candidate_and_report_outputs(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    data = repo / "data"
    scout_workflow = data / "model" / "scout_workflow"
    candidates = data / "model" / "candidates"
    reports = data / "model" / "reports"
    (reports / "holdout_compare").mkdir(parents=True, exist_ok=True)
    (reports / "low_value_contract_holdout").mkdir(parents=True, exist_ok=True)
    (reports / "candidate_promotion").mkdir(parents=True, exist_ok=True)
    scout_workflow.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)

    (scout_workflow / "future_scored_review").mkdir()
    (scout_workflow / "memos_test_20260226T153746Z").mkdir()
    (scout_workflow / "scout_workflow_summary_test_20260226T153746Z.json").write_text("{}", encoding="utf-8")
    (scout_workflow / "weekly_ops_summary_test_20260227T191530Z.json").write_text("{}", encoding="utf-8")

    (candidates / "cheap_aggressive_2024_25_promotion").mkdir()
    (candidates / "cheap_aggressive_2024_25_t60").mkdir()
    (candidates / "cheap_aggressive_prod60.csv").write_text("x", encoding="utf-8")

    (reports / "benchmark_holdouts").mkdir()
    (reports / "holdout_compare" / "no_contract_fast").mkdir()
    (reports / "holdout_compare" / "compare_full_vs_no_contract_tp.csv").write_text("x", encoding="utf-8")
    (reports / "low_value_contract_holdout" / "cheap_aggressive_drop_contract_security").mkdir()
    (reports / "candidate_promotion" / "candidate_vs_champion.json").write_text("{}", encoding="utf-8")
    (reports / "candidate_promotion" / "future_scored_candidate_vs_champion.json").write_text("{}", encoding="utf-8")

    processed = data / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest = processed / "organization_manifest.csv"
    manifest.write_text(
        "basename,league_slug,season,country,source_path,combined_path,country_path,season_path,duplicate_count,duplicate_collision\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    summary = archive_stale_workspace_artifacts(execute=True)

    archive_root = Path(str(summary["archive_root"]))
    assert not (scout_workflow / "memos_test_20260226T153746Z").exists()
    assert not (candidates / "cheap_aggressive_2024_25_promotion").exists()
    assert not (reports / "benchmark_holdouts").exists()
    assert not (reports / "candidate_promotion" / "candidate_vs_champion.json").exists()

    assert (scout_workflow / "future_scored_review").exists()
    assert (candidates / "cheap_aggressive_prod60.csv").exists()
    assert (reports / "candidate_promotion" / "future_scored_candidate_vs_champion.json").exists()

    assert (archive_root / "data/model/scout_workflow/memos_test_20260226T153746Z").exists()
    assert (archive_root / "data/model/candidates/cheap_aggressive_2024_25_promotion").exists()
    assert (archive_root / "data/model/reports/benchmark_holdouts").exists()
