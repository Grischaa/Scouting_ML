from __future__ import annotations


def assert_common_summary_contract(summary: dict) -> None:
    required_keys = {"generated_at_utc", "status", "inputs", "flags", "artifacts", "snapshots"}
    missing = required_keys.difference(summary)
    assert not missing, f"summary missing keys: {sorted(missing)}"
    assert summary["status"] == "ok"
    assert isinstance(summary["inputs"], dict)
    assert isinstance(summary["flags"], dict)
    assert isinstance(summary["artifacts"], dict)
    assert isinstance(summary["snapshots"], dict)

    for name, meta in summary["artifacts"].items():
        if meta is None:
            continue
        assert isinstance(meta, dict), f"artifact {name!r} is not a metadata dict"
        assert "path" in meta, f"artifact {name!r} missing path"
        assert "exists" in meta, f"artifact {name!r} missing exists"
