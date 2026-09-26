from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from .canonical import fingerprint as canonical_fingerprint


class ProtocolError(ValueError):
    """Raised when a research protocol or inspected configuration is invalid."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{context} must be a mapping")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ProtocolError(f"{context} has unknown fields: {', '.join(unknown)}")


def _parse_date(value: Any, context: str) -> date:
    if isinstance(value, datetime):
        raise ProtocolError(f"{context} must be a date without a time component")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ProtocolError(f"{context} must use ISO YYYY-MM-DD format") from exc
        if value != parsed.isoformat():
            raise ProtocolError(f"{context} must use canonical ISO YYYY-MM-DD format")
        return parsed
    raise ProtocolError(f"{context} must be an ISO date string or YAML date")


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    @classmethod
    def from_raw(cls, value: Any, context: str) -> DateRange:
        if isinstance(value, Mapping):
            _strict_keys(value, {"start", "end"}, context)
            if "start" not in value or "end" not in value:
                raise ProtocolError(f"{context} requires start and end")
            start_raw, end_raw = value["start"], value["end"]
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            start_raw, end_raw = value
        else:
            raise ProtocolError(f"{context} must be a two-element range or start/end mapping")

        start = _parse_date(start_raw, f"{context}.start")
        end = _parse_date(end_raw, f"{context}.end")
        if start > end:
            raise ProtocolError(
                f"{context} start {start.isoformat()} is after end {end.isoformat()}"
            )
        return cls(start=start, end=end)

    def iso_pair(self) -> tuple[str, str]:
        return self.start.isoformat(), self.end.isoformat()

    def to_manifest(self) -> list[str]:
        return list(self.iso_pair())

    def overlaps(self, other: DateRange) -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class DiscoveryStage:
    fit: DateRange
    tune: DateRange
    selection: DateRange

    @classmethod
    def from_raw(cls, value: Any, context: str = "discovery") -> DiscoveryStage:
        raw = _mapping(value, context)
        _strict_keys(raw, {"fit", "tune", "selection"}, context)
        for required in ("fit", "tune", "selection"):
            if required not in raw:
                raise ProtocolError(f"{context}.{required} is required")
        stage = cls(
            fit=DateRange.from_raw(raw["fit"], f"{context}.fit"),
            tune=DateRange.from_raw(raw["tune"], f"{context}.tune"),
            selection=DateRange.from_raw(raw["selection"], f"{context}.selection"),
        )
        _validate_ordered_ranges(
            [("fit", stage.fit), ("tune", stage.tune), ("selection", stage.selection)],
            context,
        )
        return stage

    def to_manifest(self) -> dict[str, Any]:
        return {
            "fit": self.fit.to_manifest(),
            "tune": self.tune.to_manifest(),
            "selection": self.selection.to_manifest(),
        }


@dataclass(frozen=True)
class FinalStage:
    fit: DateRange
    tune: DateRange
    evaluation: DateRange

    @classmethod
    def from_raw(cls, value: Any, context: str = "final") -> FinalStage:
        raw = _mapping(value, context)
        _strict_keys(raw, {"fit", "tune", "evaluation"}, context)
        for required in ("fit", "tune", "evaluation"):
            if required not in raw:
                raise ProtocolError(f"{context}.{required} is required")
        stage = cls(
            fit=DateRange.from_raw(raw["fit"], f"{context}.fit"),
            tune=DateRange.from_raw(raw["tune"], f"{context}.tune"),
            evaluation=DateRange.from_raw(raw["evaluation"], f"{context}.evaluation"),
        )
        _validate_ordered_ranges(
            [("fit", stage.fit), ("tune", stage.tune), ("evaluation", stage.evaluation)],
            context,
        )
        return stage

    def to_manifest(self) -> dict[str, Any]:
        return {
            "fit": self.fit.to_manifest(),
            "tune": self.tune.to_manifest(),
            "evaluation": self.evaluation.to_manifest(),
        }


def _validate_ordered_ranges(
    ranges: list[tuple[str, DateRange]],
    context: str,
) -> None:
    for (left_name, left), (right_name, right) in pairwise(ranges):
        if left.overlaps(right):
            raise ProtocolError(
                f"{context}.{left_name} and {context}.{right_name} overlap"
            )
        if left.end >= right.start:
            raise ProtocolError(
                f"{context}.{left_name} must end before {context}.{right_name} starts"
            )


_CURRENT_LABEL_CANONICAL = "Ref($close,-2)/Ref($close,-1)-1"


def _normalize_expression(expression: str) -> str:
    return "".join(expression.split())


@dataclass(frozen=True)
class LabelAvailability:
    expression: str
    supported: bool
    entry_offset_sessions: int | None = None
    endpoint_offset_sessions: int | None = None
    return_duration_sessions: int | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "supported": self.supported,
            "entry_offset_sessions": self.entry_offset_sessions,
            "endpoint_offset_sessions": self.endpoint_offset_sessions,
            "return_duration_sessions": self.return_duration_sessions,
        }


def infer_label_availability(expression: str) -> LabelAvailability:
    if not isinstance(expression, str) or not expression.strip():
        raise ProtocolError("label.expression must be a non-empty string")
    normalized = _normalize_expression(expression)
    if normalized == _CURRENT_LABEL_CANONICAL:
        return LabelAvailability(
            expression=_CURRENT_LABEL_CANONICAL,
            supported=True,
            entry_offset_sessions=1,
            endpoint_offset_sessions=2,
            return_duration_sessions=1,
        )
    return LabelAvailability(expression=normalized, supported=False)


@dataclass(frozen=True)
class ProvenanceSpec:
    dataset_snapshot_id: str | None = None
    calendar_id: str | None = None

    @classmethod
    def from_raw(cls, value: Any) -> ProvenanceSpec:
        if value is None:
            return cls()
        raw = _mapping(value, "provenance")
        _strict_keys(raw, {"dataset_snapshot_id", "calendar_id"}, "provenance")
        dataset_snapshot_id = raw.get("dataset_snapshot_id")
        calendar_id = raw.get("calendar_id")
        for field_name, field_value in (
            ("dataset_snapshot_id", dataset_snapshot_id),
            ("calendar_id", calendar_id),
        ):
            if field_value is not None and (
                not isinstance(field_value, str) or not field_value.strip()
            ):
                raise ProtocolError(f"provenance.{field_name} must be a non-empty string or null")
        return cls(
            dataset_snapshot_id=dataset_snapshot_id,
            calendar_id=calendar_id,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "calendar_id": self.calendar_id,
        }


@dataclass(frozen=True)
class ResearchProtocol:
    version: int
    name: str
    universe: str
    discovery: DiscoveryStage
    final: FinalStage
    label: LabelAvailability
    provenance: ProvenanceSpec = ProvenanceSpec()

    def to_manifest(self) -> dict[str, Any]:
        """Public, secret-free, canonicalizable protocol representation."""
        return {
            "schema": "quantaalpha.research_protocol.v1",
            "version": self.version,
            "name": self.name,
            "universe": self.universe,
            "discovery": self.discovery.to_manifest(),
            "final": self.final.to_manifest(),
            "label": self.label.to_manifest(),
            "provenance": self.provenance.to_manifest(),
        }

    def semantic_manifest(self) -> dict[str, Any]:
        """Fields that can be compared with current Qlib configuration."""
        return {
            "version": self.version,
            "universe": self.universe,
            "discovery": self.discovery.to_manifest(),
            "final": self.final.to_manifest(),
            "label": self.label.to_manifest(),
        }

    def fingerprint(self) -> str:
        """Identity of the complete declared/public protocol manifest."""
        return canonical_fingerprint(self.to_manifest())

    def semantic_fingerprint(self) -> str:
        """Identity of fields that can be compared with effective Qlib config."""
        return canonical_fingerprint(self.semantic_manifest())


def protocol_from_mapping(value: Any, *, source: str = "protocol") -> ResearchProtocol:
    raw = _mapping(value, source)
    _strict_keys(
        raw,
        {"version", "name", "universe", "discovery", "final", "label", "provenance"},
        source,
    )
    for required in ("version", "name", "universe", "discovery", "final", "label"):
        if required not in raw:
            raise ProtocolError(f"{source}.{required} is required")

    version = raw["version"]
    if type(version) is not int or version != 1:
        raise ProtocolError(f"{source}.version must be integer 1")

    name = raw["name"]
    universe = raw["universe"]
    if not isinstance(name, str) or not name.strip():
        raise ProtocolError(f"{source}.name must be a non-empty string")
    if not isinstance(universe, str) or not universe.strip():
        raise ProtocolError(f"{source}.universe must be a non-empty string")

    label_raw = _mapping(raw["label"], f"{source}.label")
    _strict_keys(
        label_raw,
        {
            "expression",
            "entry_offset_sessions",
            "endpoint_offset_sessions",
            "return_duration_sessions",
        },
        f"{source}.label",
    )
    if "expression" not in label_raw:
        raise ProtocolError(f"{source}.label.expression is required")
    label = infer_label_availability(label_raw["expression"])

    declared_offsets = {
        "entry_offset_sessions": label_raw.get("entry_offset_sessions"),
        "endpoint_offset_sessions": label_raw.get("endpoint_offset_sessions"),
        "return_duration_sessions": label_raw.get("return_duration_sessions"),
    }
    if label.supported:
        for field_name, declared in declared_offsets.items():
            inferred = getattr(label, field_name)
            if declared is not None and declared != inferred:
                raise ProtocolError(
                    f"{source}.label.{field_name}={declared!r} disagrees with "
                    f"supported label semantics {inferred!r}"
                )
    elif any(value is not None for value in declared_offsets.values()):
        raise ProtocolError(
            f"{source}.label expression is unsupported; availability offsets "
            "cannot be asserted by this protocol version"
        )

    return ResearchProtocol(
        version=version,
        name=name.strip(),
        universe=universe.strip(),
        discovery=DiscoveryStage.from_raw(raw["discovery"], f"{source}.discovery"),
        final=FinalStage.from_raw(raw["final"], f"{source}.final"),
        label=label,
        provenance=ProvenanceSpec.from_raw(raw.get("provenance")),
    )


def load_protocol(path: str | Path) -> ResearchProtocol:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return protocol_from_mapping(raw, source="protocol")
