"""
QuantaAlpha CLI entry.

Commands:
  quantaalpha mine         - run factor mining
  quantaalpha backtest     - run backtest
  quantaalpha protocol     - inspect research evaluation roles/provenance
  quantaalpha health_check - environment health check
"""

from __future__ import annotations

import sys
from pathlib import Path


def _run_protocol_cli() -> int:
    # Keep protocol preflight independent of mining/Qlib/LLM imports.
    from quantaalpha.evaluation.cli import main as protocol_main

    return protocol_main(sys.argv[2:])


def _protocol_help_entry():
    """Research protocol preflight; run `quantaalpha protocol --help` for options."""
    return _run_protocol_cli()


def _load_legacy_commands():
    from dotenv import load_dotenv

    # Preserve the existing .env behavior for mining/backtest/utility commands.
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv(".env")

    from quantaalpha.app.utils.health_check import health_check
    from quantaalpha.app.utils.info import collect_info
    from quantaalpha.pipeline.factor_backtest import main as backtest
    from quantaalpha.pipeline.factor_mining import main as mine

    return {
        "mine": mine,
        "backtest": backtest,
        "protocol": _protocol_help_entry,
        "health_check": health_check,
        "collect_info": collect_info,
    }


def app():
    if len(sys.argv) > 1 and sys.argv[1] == "protocol":
        return _run_protocol_cli()

    import fire

    return fire.Fire(_load_legacy_commands())


if __name__ == "__main__":
    app()
