from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from quantaalpha.evaluation.adapters.qlib_config import inspect_configs
from quantaalpha.evaluation.preflight import ProtocolMismatchError, run_preflight
from quantaalpha.evaluation.protocol import (
    ProtocolError,
    infer_label_availability,
    load_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "quantaalpha/factors/factor_template/conf_baseline.yaml"
COMBINED = ROOT / "quantaalpha/factors/factor_template/conf_combined_factors.yaml"
BACKTEST = ROOT / "configs/backtest.yaml"


PROTOCOL_TEXT = """
version: 1
name: current-two-stage
universe: csi300
discovery:
  fit: ["2016-01-01", "2019-12-31"]
  tune: ["2020-01-01", "2020-12-31"]
  selection: ["2021-01-01", "2021-12-31"]
final:
  fit: ["2016-01-01", "2020-12-31"]
  tune: ["2021-01-01", "2021-12-31"]
  evaluation: ["2022-01-01", "2025-12-26"]
label:
  expression: "Ref($close, -2) / Ref($close, -1) - 1"
  entry_offset_sessions: 1
  endpoint_offset_sessions: 2
  return_duration_sessions: 1
provenance:
  dataset_snapshot_id: null
  calendar_id: null
"""


def write_protocol(tmp_path: Path, text: str = PROTOCOL_TEXT) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "research_protocol.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_quoted_and_yaml_date_objects_have_identical_manifest_and_hash(tmp_path):
    quoted = load_protocol(write_protocol(tmp_path / "quoted"))

    unquoted_text = PROTOCOL_TEXT
    for value in (
        "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31",
        "2021-01-01", "2021-12-31", "2022-01-01", "2025-12-26",
    ):
        unquoted_text = unquoted_text.replace(f'"{value}"', value)
    unquoted = load_protocol(write_protocol(tmp_path / "unquoted", unquoted_text))

    assert quoted.to_manifest() == unquoted.to_manifest()
    assert quoted.fingerprint() == unquoted.fingerprint()
    assert isinstance(yaml.safe_load(unquoted_text)["discovery"]["fit"][0], date)


def test_mapping_order_does_not_change_fingerprint(tmp_path):
    original = load_protocol(write_protocol(tmp_path / "one"))
    raw = yaml.safe_load(PROTOCOL_TEXT)
    reordered = {
        "label": raw["label"],
        "final": raw["final"],
        "version": raw["version"],
        "provenance": raw["provenance"],
        "discovery": raw["discovery"],
        "universe": raw["universe"],
        "name": raw["name"],
    }
    path = tmp_path / "two" / "research_protocol.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")
    loaded = load_protocol(path)

    assert loaded.to_manifest() == original.to_manifest()
    assert loaded.fingerprint() == original.fingerprint()


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda raw: raw["discovery"].__setitem__("fit", ["2020-01-01", "2019-12-31"]), "start"),
        (lambda raw: raw["discovery"].__setitem__("tune", ["2019-12-01", "2020-12-31"]), "overlap"),
        (lambda raw: raw["discovery"].pop("selection"), "selection"),
        (lambda raw: raw["final"].__setitem__("evaluation", []), "evaluation"),
    ],
)
def test_invalid_ranges_fail_clearly(tmp_path, mutator, match):
    raw = yaml.safe_load(PROTOCOL_TEXT)
    mutator(raw)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match=match):
        load_protocol(path)


def test_unknown_top_level_fields_are_rejected(tmp_path):
    raw = yaml.safe_load(PROTOCOL_TEXT)
    raw["api_key"] = "do-not-serialize-me"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="unknown"):
        load_protocol(path)


def test_supported_label_reports_availability():
    info = infer_label_availability("Ref($close, -2) / Ref($close, -1) - 1")
    assert info.supported is True
    assert info.entry_offset_sessions == 1
    assert info.endpoint_offset_sessions == 2
    assert info.return_duration_sessions == 1


def test_unknown_label_is_reported_unsupported_without_guessing():
    info = infer_label_availability("Ref($close, -5) / Ref($close, -1) - 1")
    assert info.supported is False
    assert info.entry_offset_sessions is None
    assert info.endpoint_offset_sessions is None
    assert info.return_duration_sessions is None


def test_current_configs_resolve_expected_research_roles():
    inspected = inspect_configs([BASELINE, COMBINED], BACKTEST)

    assert inspected.protocol.discovery.fit.iso_pair() == ("2016-01-01", "2019-12-31")
    assert inspected.protocol.discovery.tune.iso_pair() == ("2020-01-01", "2020-12-31")
    assert inspected.protocol.discovery.selection.iso_pair() == ("2021-01-01", "2021-12-31")
    assert inspected.protocol.final.fit.iso_pair() == ("2016-01-01", "2020-12-31")
    assert inspected.protocol.final.tune.iso_pair() == ("2021-01-01", "2021-12-31")
    assert inspected.protocol.final.evaluation.iso_pair() == ("2022-01-01", "2025-12-26")
    assert inspected.protocol.universe == "csi300"
    assert inspected.protocol.label.supported is True
    assert inspected.mining_visible_range.iso_pair() == ("2016-01-01", "2025-12-26")
    assert any("does not establish data isolation" in warning for warning in inspected.warnings)


def test_baseline_and_combined_templates_must_agree(tmp_path):
    raw = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    raw["task"]["dataset"]["kwargs"]["segments"]["test"] = ["2021-02-01", "2021-12-31"]
    raw["port_analysis_config"]["backtest"]["start_time"] = "2021-02-01"
    bad = tmp_path / "conf_bad.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="mining configurations disagree"):
        inspect_configs([BASELINE, bad], BACKTEST)


def test_ambiguous_mining_labels_fail_instead_of_choosing_first(tmp_path):
    raw = yaml.safe_load(COMBINED.read_text(encoding="utf-8"))
    dataloaders = raw["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["dataloader_l"]
    dataloaders.append(
        {
            "class": "qlib.contrib.data.loader.QlibDataLoader",
            "kwargs": {
                "config": {
                    "label": [
                        ["Ref($close, -3) / Ref($close, -1) - 1"],
                        ["OTHER_LABEL"],
                    ]
                }
            },
        }
    )
    bad = tmp_path / "conf_ambiguous.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="ambiguous label expressions"):
        inspect_configs([bad], BACKTEST)


def test_importing_top_level_cli_does_not_eagerly_import_mining():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import quantaalpha.cli; "
                "print('quantaalpha.pipeline.factor_mining' in sys.modules); "
                "print('quantaalpha.pipeline.factor_backtest' in sys.modules)"
            ),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["False", "False"]


def test_legacy_command_map_preserves_existing_entries(monkeypatch):
    from quantaalpha import cli

    dotenv = ModuleType("dotenv")
    dotenv_calls = []
    dotenv.load_dotenv = lambda path: dotenv_calls.append(str(path))
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)

    expected = {}
    modules = {
        "quantaalpha.app.utils.health_check": ("health_check", lambda: None),
        "quantaalpha.app.utils.info": ("collect_info", lambda: None),
        "quantaalpha.pipeline.factor_backtest": ("main", lambda: None),
        "quantaalpha.pipeline.factor_mining": ("main", lambda: None),
    }
    for module_name, (attribute, value) in modules.items():
        module = ModuleType(module_name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, module_name, module)
        expected[module_name] = value

    commands = cli._load_legacy_commands()

    assert commands["mine"] is expected["quantaalpha.pipeline.factor_mining"]
    assert commands["backtest"] is expected["quantaalpha.pipeline.factor_backtest"]
    assert commands["health_check"] is expected["quantaalpha.app.utils.health_check"]
    assert commands["collect_info"] is expected["quantaalpha.app.utils.info"]
    assert commands["protocol"] is cli._protocol_help_entry
    assert len(dotenv_calls) == 1


def test_opted_in_protocol_mismatch_is_an_error(tmp_path):
    raw = yaml.safe_load(PROTOCOL_TEXT)
    raw["final"]["evaluation"] = ["2023-01-01", "2025-12-26"]
    protocol_path = tmp_path / "mismatch.yaml"
    protocol_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolMismatchError, match="final.evaluation"):
        run_preflight([BASELINE, COMBINED], BACKTEST, protocol_path=protocol_path)


def test_missing_protocol_preserves_inspection_only_mode():
    report = run_preflight([BASELINE, COMBINED], BACKTEST, protocol_path=None)

    assert report.mode == "inspect-only"
    assert report.validated_against_declared_protocol is False
    assert report.guarantees["configuration_validated"] is True
    assert report.guarantees["data_isolation_enforced"] is False
    assert report.guarantees["operator_causality_certified"] is False
    assert report.guarantees["final_period_unobserved_certified"] is False


def test_matching_protocol_validates_and_manifest_contains_no_secret_config(tmp_path):
    path = write_protocol(tmp_path)
    report = run_preflight([BASELINE, COMBINED], BACKTEST, protocol_path=path)
    payload = json.dumps(report.to_dict(), sort_keys=True)

    assert report.mode == "validated"
    assert report.validated_against_declared_protocol is True
    assert report.protocol_hash == load_protocol(path).fingerprint()
    assert report.semantic_hash == report.effective_semantic_hash
    assert "provider_uri" not in payload
    assert "api_key" not in payload
    assert "secret" not in payload.lower()


def test_calendar_dependent_purge_is_not_invented(tmp_path):
    report = run_preflight(
        [BASELINE, COMBINED],
        BACKTEST,
        protocol_path=write_protocol(tmp_path),
    )

    assert report.protocol.provenance.calendar_id is None
    assert report.label_boundary_status == "calendar-required"
    assert any("calendar" in warning.lower() for warning in report.warnings)


def test_cli_protocol_inspect_avoids_mining_imports_and_network(tmp_path):
    protocol = write_protocol(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    command = [
        sys.executable,
        "-m",
        "quantaalpha.cli",
        "protocol",
        "inspect",
        "--mining-config",
        str(BASELINE),
        "--mining-config",
        str(COMBINED),
        "--backtest-config",
        str(BACKTEST),
        "--protocol",
        str(protocol),
        "--format",
        "json",
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "validated"
    assert payload["protocol"]["discovery"]["selection"] == ["2021-01-01", "2021-12-31"]
    assert payload["protocol"]["final"]["evaluation"] == ["2022-01-01", "2025-12-26"]
    assert payload["guarantees"]["data_isolation_enforced"] is False


def test_preview_requires_explicit_output_directory_and_writes_only_there(tmp_path):
    protocol = write_protocol(tmp_path / "input")
    output = tmp_path / "preview"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    command = [
        sys.executable,
        "-m",
        "quantaalpha.cli",
        "protocol",
        "preview",
        "--mining-config",
        str(BASELINE),
        "--backtest-config",
        str(BACKTEST),
        "--protocol",
        str(protocol),
        "--output-dir",
        str(output),
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in output.iterdir()) == [
        "research_protocol_manifest.json",
        "research_protocol_report.json",
    ]
    report = json.loads((output / "research_protocol_report.json").read_text(encoding="utf-8"))
    assert report["guarantees"]["configuration_validated"] is True
    assert report["guarantees"]["data_isolation_enforced"] is False
