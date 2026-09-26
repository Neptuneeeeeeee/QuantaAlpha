from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
FUNCTION_LIB = ROOT / "quantaalpha/factors/coder/function_lib.py"
EXPR_PARSER = ROOT / "quantaalpha/factors/coder/expr_parser.py"


@pytest.fixture(scope="module")
def function_lib():
    spec = importlib.util.spec_from_file_location("_temporal_function_lib_test", FUNCTION_LIB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def expr_parser():
    spec = importlib.util.spec_from_file_location("_temporal_expr_parser_test", EXPR_PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def panel():
    dates = pd.date_range("2021-12-27", periods=6, freq="B")
    instruments = ["AAA", "BBB"]
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=["datetime", "instrument"],
    )
    values = np.arange(1, len(index) + 1, dtype=float)
    return pd.DataFrame({"$close": values}, index=index)


def perturb_future(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    changed = panel.copy()
    future = changed.index.get_level_values("datetime") > cutoff
    changed.loc[future, "$close"] *= -1000.0
    return changed


@pytest.mark.parametrize("name", ["DELTA", "DELAY", "TS_PCTCHANGE"])
@pytest.mark.parametrize("period", [-1, -2])
def test_negative_lag_periods_are_rejected(function_lib, panel, name, period):
    operator = getattr(function_lib, name)

    with pytest.raises(ValueError, match=name):
        operator(panel, period)


@pytest.mark.parametrize("name", ["DELTA", "DELAY", "TS_PCTCHANGE"])
@pytest.mark.parametrize("period", [1.5, True])
def test_non_integer_lag_periods_are_rejected(function_lib, panel, name, period):
    operator = getattr(function_lib, name)

    with pytest.raises(ValueError, match="integer"):
        operator(panel, period)


@pytest.mark.parametrize("name", ["DELTA", "DELAY", "TS_PCTCHANGE"])
def test_zero_period_remains_causal_and_supported(function_lib, panel, name):
    operator = getattr(function_lib, name)

    result = operator(panel, 0)

    assert result.index.equals(panel.index)
    assert result.notna().all().all()


@pytest.mark.parametrize("name", ["DELTA", "DELAY", "TS_PCTCHANGE"])
def test_positive_period_matches_previous_pandas_semantics(function_lib, panel, name):
    period = 2
    result = getattr(function_lib, name)(panel, period)

    grouped = panel.groupby("instrument")
    if name == "DELTA":
        expected = grouped.transform(lambda x: x.diff(periods=period))
    elif name == "DELAY":
        expected = grouped.transform(lambda x: x.shift(period))
    else:
        expected = grouped.transform(
            lambda x: x.pct_change(periods=period, fill_method=None).fillna(0)
        )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("name", ["DELTA", "DELAY", "TS_PCTCHANGE"])
@pytest.mark.parametrize("period", [1, 2, np.int64(2)])
def test_future_perturbation_does_not_change_historical_prefix(
    function_lib,
    panel,
    name,
    period,
):
    operator = getattr(function_lib, name)
    cutoff = pd.Timestamp("2021-12-30")
    changed = perturb_future(panel, cutoff)

    original_result = operator(panel, period)
    changed_result = operator(changed, period)

    historical = panel.index.get_level_values("datetime") <= cutoff
    assert_frame_equal(
        original_result.loc[historical],
        changed_result.loc[historical],
        check_dtype=False,
    )


def test_negative_delta_and_pctchange_would_read_future_without_contract(
    function_lib,
    panel,
):
    """Document why negative lag periods are a temporal-safety concern."""
    cutoff = pd.Timestamp("2021-12-30")
    changed = perturb_future(panel, cutoff)

    for name in ("DELTA", "TS_PCTCHANGE"):
        # Use the underlying pandas operation to record the pre-fix behavior
        # without bypassing the production guard once it exists.
        grouped_original = panel.groupby("instrument")["$close"]
        grouped_changed = changed.groupby("instrument")["$close"]
        if name == "DELTA":
            before = grouped_original.transform(lambda x: x.diff(periods=-1))
            after = grouped_changed.transform(lambda x: x.diff(periods=-1))
        else:
            before = grouped_original.transform(
                lambda x: x.pct_change(periods=-1, fill_method=None).fillna(0)
            )
            after = grouped_changed.transform(
                lambda x: x.pct_change(periods=-1, fill_method=None).fillna(0)
            )

        historical = panel.index.get_level_values("datetime") <= cutoff
        assert not before.loc[historical].equals(after.loc[historical])


@pytest.mark.parametrize("name", ["DELTA", "DELAY", "TS_PCTCHANGE"])
def test_expression_parser_path_is_still_guarded(
    function_lib,
    expr_parser,
    panel,
    name,
):
    expression = f"{name}($close, -1)"
    parsed = expr_parser.parse_symbol(expression, panel.columns)
    parsed = expr_parser.parse_expression(parsed)
    for column in panel.columns:
        parsed = parsed.replace(column[1:], f"df[{column!r}]")

    with pytest.raises(ValueError, match=name):
        eval(parsed, {"df": panel, name: getattr(function_lib, name)})


def test_lag_guards_survive_python_optimized_mode():
    code = f"""
import importlib.util
import pandas as pd
spec = importlib.util.spec_from_file_location("_optimized_temporal_test", {str(FUNCTION_LIB)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
idx = pd.MultiIndex.from_product(
    [pd.date_range("2021-01-01", periods=3), ["AAA"]],
    names=["datetime", "instrument"],
)
df = pd.DataFrame({{"$close": [1.0, 2.0, 3.0]}}, index=idx)
for name in ("DELTA", "DELAY", "TS_PCTCHANGE"):
    try:
        getattr(m, name)(df, -1)
    except ValueError:
        continue
    raise SystemExit(2)
raise SystemExit(0)
"""
    proc = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
