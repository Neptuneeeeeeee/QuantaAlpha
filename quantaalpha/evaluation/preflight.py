from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.qlib_config import InspectedConfigs, inspect_configs
from .protocol import ProtocolError, ResearchProtocol, load_protocol


class ProtocolMismatchError(ProtocolError):
    """Raised when an opted-in declaration disagrees with effective config."""


def _find_differences(
    declared: Any,
    effective: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(declared, dict) and isinstance(effective, dict):
        differences: list[str] = []
        for key in sorted(set(declared) | set(effective)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in declared or key not in effective:
                differences.append(path)
                continue
            differences.extend(
                _find_differences(declared[key], effective[key], path)
            )
        return differences
    if declared != effective:
        return [prefix or "<root>"]
    return []


@dataclass(frozen=True)
class PreflightReport:
    protocol: ResearchProtocol
    protocol_hash: str
    semantic_hash: str
    effective_semantic_hash: str
    mode: str
    validated_against_declared_protocol: bool
    label_boundary_status: str
    guarantees: dict[str, bool]
    warnings: tuple[str, ...]
    inspected: InspectedConfigs

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "quantaalpha.research_protocol_report.v1",
            "mode": self.mode,
            "validated_against_declared_protocol": self.validated_against_declared_protocol,
            "protocol_hash": self.protocol_hash,
            "semantic_hash": self.semantic_hash,
            "effective_semantic_hash": self.effective_semantic_hash,
            "protocol": self.protocol.to_manifest(),
            "effective_visibility": {
                "mining": self.inspected.mining_visible_range.to_manifest(),
                "final": self.inspected.final_visible_range.to_manifest(),
            },
            "label_boundary_status": self.label_boundary_status,
            "guarantees": dict(self.guarantees),
            "warnings": list(self.warnings),
            "sources": {
                "mining_configs": list(self.inspected.mining_configs),
                "backtest_config": self.inspected.backtest_config,
            },
        }

    def write_preview(self, output_dir: str | Path) -> tuple[Path, Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "research_protocol_manifest.json"
        report_path = destination / "research_protocol_report.json"
        manifest_path.write_text(
            json.dumps(self.protocol.to_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path, report_path


def _label_boundary_status(protocol: ResearchProtocol) -> str:
    if not protocol.label.supported:
        return "unsupported-label"
    if protocol.provenance.calendar_id is None:
        return "calendar-required"
    return "calendar-declared-not-verified"


def run_preflight(
    mining_configs: list[str | Path],
    backtest_config: str | Path,
    *,
    protocol_path: str | Path | None = None,
) -> PreflightReport:
    inspected = inspect_configs(mining_configs, backtest_config)
    effective = inspected.protocol
    warnings = list(inspected.warnings)

    if protocol_path is None:
        protocol = effective
        mode = "inspect-only"
        validated = False
    else:
        protocol = load_protocol(protocol_path)
        differences = _find_differences(
            protocol.semantic_manifest(),
            effective.semantic_manifest(),
        )
        if differences:
            raise ProtocolMismatchError(
                "declared protocol disagrees with effective configuration at: "
                + ", ".join(differences)
            )
        mode = "validated"
        validated = True
        if protocol.provenance.dataset_snapshot_id is not None:
            warnings.append(
                "dataset_snapshot_id is declared but this preflight does not verify "
                "the backing market-data bytes."
            )
        if protocol.provenance.calendar_id is not None:
            warnings.append(
                "calendar_id is declared but this preflight does not load or verify "
                "the trading calendar."
            )

    label_status = _label_boundary_status(protocol)
    if label_status == "calendar-required":
        warnings.append(
            "Supported label availability is session-based, but no versioned calendar "
            "is declared; purge/embargo rows cannot be resolved by preflight."
        )
    elif label_status == "unsupported-label":
        warnings.append(
            "Label expression is not supported by protocol v1; no availability horizon "
            "was inferred."
        )

    guarantees = {
        "configuration_validated": True,
        "declared_protocol_matches_effective_config": validated,
        "data_isolation_enforced": False,
        "operator_causality_certified": False,
        "final_period_unobserved_certified": False,
        "dataset_snapshot_verified": False,
        "trading_calendar_verified": False,
    }

    return PreflightReport(
        protocol=protocol,
        protocol_hash=protocol.fingerprint(),
        semantic_hash=protocol.semantic_fingerprint(),
        effective_semantic_hash=effective.semantic_fingerprint(),
        mode=mode,
        validated_against_declared_protocol=validated,
        label_boundary_status=label_status,
        guarantees=guarantees,
        warnings=tuple(dict.fromkeys(warnings)),
        inspected=inspected,
    )
