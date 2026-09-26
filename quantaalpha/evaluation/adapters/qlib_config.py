from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..protocol import (
    DateRange,
    DiscoveryStage,
    FinalStage,
    ProtocolError,
    ProvenanceSpec,
    ResearchProtocol,
    infer_label_availability,
)


@dataclass(frozen=True)
class InspectedConfigs:
    protocol: ResearchProtocol
    mining_visible_range: DateRange
    final_visible_range: DateRange
    warnings: tuple[str, ...]
    mining_configs: tuple[str, ...]
    backtest_config: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_manifest(),
            "mining_visible_range": self.mining_visible_range.to_manifest(),
            "final_visible_range": self.final_visible_range.to_manifest(),
            "warnings": list(self.warnings),
            "sources": {
                "mining_configs": list(self.mining_configs),
                "backtest_config": self.backtest_config,
            },
        }


@dataclass(frozen=True)
class _MiningSnapshot:
    discovery: DiscoveryStage
    universe: str
    label_expression: str
    visible_range: DateRange
    score_range: DateRange


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ProtocolError(f"failed to parse YAML {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError(f"{path} must contain a YAML mapping")
    return raw


def _get(mapping: Mapping[str, Any], keys: Iterable[str], context: str) -> Any:
    current: Any = mapping
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise ProtocolError(f"{context} missing {'.'.join(walked)}")
        current = current[key]
    return current


def _label_expressions(node: Any) -> list[str]:
    found: list[str] = []
    if isinstance(node, Mapping):
        if "label" in node:
            value = node["label"]
            # Qlib config convention: [[expr...], [name...]]
            if (
                isinstance(value, (list, tuple))
                and value
                and isinstance(value[0], (list, tuple))
                and value[0]
                and isinstance(value[0][0], str)
            ):
                found.append(value[0][0])
            elif isinstance(value, str):
                found.append(value)
        for child in node.values():
            found.extend(_label_expressions(child))
    elif isinstance(node, (list, tuple)):
        for child in node:
            found.extend(_label_expressions(child))
    return found


def _resolve_mining_label(node: Any, path: Path) -> str:
    expressions = _label_expressions(node)
    if not expressions:
        raise ProtocolError(f"mining config {path} label expression could not be resolved")
    semantic = {infer_label_availability(expr).expression for expr in expressions}
    if len(semantic) != 1:
        raise ProtocolError(
            f"mining config {path} has ambiguous label expressions: "
            + ", ".join(sorted(semantic))
        )
    return expressions[0]


def _mining_snapshot(path: Path) -> _MiningSnapshot:
    raw = _read_yaml(path)
    segments = _get(
        raw,
        ("task", "dataset", "kwargs", "segments"),
        f"mining config {path}",
    )
    if not isinstance(segments, Mapping):
        raise ProtocolError(f"mining config {path} task.dataset.kwargs.segments must be a mapping")

    discovery = DiscoveryStage.from_raw(
        {
            "fit": segments.get("train"),
            "tune": segments.get("valid"),
            "selection": segments.get("test"),
        },
        f"mining config {path}.segments",
    )

    handler = _get(
        raw,
        ("task", "dataset", "kwargs", "handler", "kwargs"),
        f"mining config {path}",
    )
    if not isinstance(handler, Mapping):
        raise ProtocolError(f"mining config {path} handler kwargs must be a mapping")

    visible_range = DateRange.from_raw(
        [handler.get("start_time"), handler.get("end_time")],
        f"mining config {path}.handler_range",
    )

    label_expression = _resolve_mining_label(handler.get("data_loader"), path)

    universe = raw.get("market")
    if not isinstance(universe, str) or not universe.strip():
        raise ProtocolError(f"mining config {path} market must be a non-empty string")

    port_backtest = _get(raw, ("port_analysis_config", "backtest"), f"mining config {path}")
    if not isinstance(port_backtest, Mapping):
        raise ProtocolError(f"mining config {path} port_analysis_config.backtest must be a mapping")
    score_range = DateRange.from_raw(
        [port_backtest.get("start_time"), port_backtest.get("end_time")],
        f"mining config {path}.portfolio_score_range",
    )
    if score_range != discovery.selection:
        raise ProtocolError(
            f"mining config {path} Qlib test segment disagrees with portfolio scoring range"
        )

    return _MiningSnapshot(
        discovery=discovery,
        universe=universe.strip(),
        label_expression=label_expression,
        visible_range=visible_range,
        score_range=score_range,
    )


def _assert_mining_agreement(
    snapshots: list[tuple[Path, _MiningSnapshot]],
) -> _MiningSnapshot:
    if not snapshots:
        raise ProtocolError("at least one mining configuration is required")
    first_path, first = snapshots[0]
    for path, snapshot in snapshots[1:]:
        fields = {
            "segments": snapshot.discovery == first.discovery,
            "universe": snapshot.universe == first.universe,
            "label": infer_label_availability(snapshot.label_expression)
            == infer_label_availability(first.label_expression),
            "handler_range": snapshot.visible_range == first.visible_range,
            "portfolio_score_range": snapshot.score_range == first.score_range,
        }
        disagreement = [name for name, same in fields.items() if not same]
        if disagreement:
            raise ProtocolError(
                "mining configurations disagree "
                f"({first_path} vs {path}) on: {', '.join(disagreement)}"
            )
    return first


def inspect_configs(
    mining_configs: Iterable[str | Path],
    backtest_config: str | Path,
) -> InspectedConfigs:
    mining_paths = [Path(path) for path in mining_configs]
    snapshots = [(path, _mining_snapshot(path)) for path in mining_paths]
    mining = _assert_mining_agreement(snapshots)

    backtest_path = Path(backtest_config)
    raw = _read_yaml(backtest_path)
    dataset = _get(raw, ("dataset",), f"backtest config {backtest_path}")
    if not isinstance(dataset, Mapping):
        raise ProtocolError(f"backtest config {backtest_path} dataset must be a mapping")
    segments = dataset.get("segments")
    if not isinstance(segments, Mapping):
        raise ProtocolError(f"backtest config {backtest_path} dataset.segments must be a mapping")

    final = FinalStage.from_raw(
        {
            "fit": segments.get("train"),
            "tune": segments.get("valid"),
            "evaluation": segments.get("test"),
        },
        f"backtest config {backtest_path}.segments",
    )

    label_expression = dataset.get("label")
    if not isinstance(label_expression, str) or not label_expression.strip():
        raise ProtocolError(f"backtest config {backtest_path} dataset.label must be a string")

    mining_label = infer_label_availability(mining.label_expression)
    final_label = infer_label_availability(label_expression)
    if mining_label != final_label:
        raise ProtocolError(
            "mining and final backtest label semantics disagree: "
            f"{mining_label.expression!r} vs {final_label.expression!r}"
        )

    data = _get(raw, ("data",), f"backtest config {backtest_path}")
    if not isinstance(data, Mapping):
        raise ProtocolError(f"backtest config {backtest_path} data must be a mapping")
    universe = data.get("market")
    if not isinstance(universe, str) or not universe.strip():
        raise ProtocolError(f"backtest config {backtest_path} data.market must be a non-empty string")
    universe = universe.strip()
    if universe != mining.universe:
        raise ProtocolError(
            f"mining universe {mining.universe!r} disagrees with final universe {universe!r}"
        )
    final_visible_range = DateRange.from_raw(
        [data.get("start_time"), data.get("end_time")],
        f"backtest config {backtest_path}.data_range",
    )

    bt = _get(raw, ("backtest", "backtest"), f"backtest config {backtest_path}")
    if not isinstance(bt, Mapping):
        raise ProtocolError(f"backtest config {backtest_path} backtest.backtest must be a mapping")
    portfolio_range = DateRange.from_raw(
        [bt.get("start_time"), bt.get("end_time")],
        f"backtest config {backtest_path}.portfolio_range",
    )
    if portfolio_range != final.evaluation:
        raise ProtocolError(
            f"backtest config {backtest_path} test segment disagrees with portfolio range"
        )

    protocol = ResearchProtocol(
        version=1,
        name="effective-config",
        universe=universe,
        discovery=mining.discovery,
        final=final,
        label=final_label,
        provenance=ProvenanceSpec(),
    )

    warnings: list[str] = []
    if mining.visible_range.end > mining.discovery.selection.end:
        warnings.append(
            "Mining handler visibility extends beyond adaptive selection; the configured "
            "scoring range does not establish data isolation."
        )
    if final_visible_range.start > final.fit.start or final_visible_range.end < final.evaluation.end:
        warnings.append(
            "Final data handler coverage does not fully contain declared fit/evaluation periods."
        )
    warnings.append(
        "No dataset snapshot identity is declared; configuration inspection does not "
        "verify the exact market-data snapshot."
    )
    warnings.append(
        "No versioned trading calendar is declared; session-aware label purging remains unresolved."
    )

    return InspectedConfigs(
        protocol=protocol,
        mining_visible_range=mining.visible_range,
        final_visible_range=final_visible_range,
        warnings=tuple(warnings),
        mining_configs=tuple(str(path) for path in mining_paths),
        backtest_config=str(backtest_path),
    )
