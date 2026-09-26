from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .preflight import ProtocolMismatchError, run_preflight
from .protocol import ProtocolError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MINING_CONFIGS = [
    _PROJECT_ROOT / "quantaalpha/factors/factor_template/conf_baseline.yaml",
    _PROJECT_ROOT / "quantaalpha/factors/factor_template/conf_combined_factors.yaml",
]
_DEFAULT_BACKTEST_CONFIG = _PROJECT_ROOT / "configs/backtest.yaml"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mining-config",
        action="append",
        dest="mining_configs",
        help=(
            "Preliminary mining Qlib YAML. Repeat to require multiple templates "
            "to agree. Defaults to the baseline and combined templates."
        ),
    )
    parser.add_argument(
        "--backtest-config",
        default=str(_DEFAULT_BACKTEST_CONFIG),
        help="Independent backtest YAML (default: configs/backtest.yaml).",
    )
    parser.add_argument(
        "--protocol",
        help=(
            "Optional ResearchProtocol YAML. When omitted, inspect effective "
            "configuration without changing legacy behavior."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantaalpha protocol",
        description=(
            "Inspect research data roles and validate an opt-in ResearchProtocol. "
            "This command validates configuration only; it does not enforce data isolation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser(
        "inspect",
        help="Inspect effective research roles and optionally validate a protocol.",
    )
    _add_common_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )

    preview_parser = sub.add_parser(
        "preview",
        help="Write the canonical protocol manifest and preflight report.",
    )
    _add_common_arguments(preview_parser)
    preview_parser.add_argument(
        "--output-dir",
        required=True,
        help="Explicit destination directory for the two preview JSON files.",
    )
    return parser


def _mining_configs(values: list[str] | None) -> list[Path]:
    return [Path(value) for value in values] if values else list(_DEFAULT_MINING_CONFIGS)


def _text_report(payload: dict) -> str:
    protocol = payload["protocol"]
    discovery = protocol["discovery"]
    final = protocol["final"]
    lines = [
        f"Mode: {payload['mode']}",
        f"Protocol hash: {payload['protocol_hash']}",
        "",
        "Discovery:",
        f"  fit:       {discovery['fit'][0]} -> {discovery['fit'][1]}",
        f"  tune:      {discovery['tune'][0]} -> {discovery['tune'][1]}",
        f"  selection: {discovery['selection'][0]} -> {discovery['selection'][1]}",
        "Final:",
        f"  fit:        {final['fit'][0]} -> {final['fit'][1]}",
        f"  tune:       {final['tune'][0]} -> {final['tune'][1]}",
        f"  evaluation: {final['evaluation'][0]} -> {final['evaluation'][1]}",
        "",
        "Guarantees:",
    ]
    for key, value in payload["guarantees"].items():
        lines.append(f"  {key}: {str(value).lower()}")
    if payload["warnings"]:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in payload["warnings"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    mining = _mining_configs(args.mining_configs)

    try:
        report = run_preflight(
            mining,
            Path(args.backtest_config),
            protocol_path=Path(args.protocol) if args.protocol else None,
        )
    except (ProtocolError, ProtocolMismatchError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if args.command == "inspect":
        payload = report.to_dict()
        if args.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_text_report(payload))
        return 0

    report.write_preview(Path(args.output_dir))
    print(f"Protocol preview written to: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
