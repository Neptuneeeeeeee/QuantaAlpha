"""Regression tests for issue #27; no market data or LLM calls are needed.

The unit tests exercise the real runner and its nested handler, replacing only
Qlib's construction-time dependencies. The final test uses real DatasetH when
pyqlib is installed and is explicitly skipped otherwise.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def runner_class():
    # Load this standalone module without importing the other backtest tools.
    path = Path(__file__).resolve().parents[2] / "quantaalpha/backtest/runner.py"
    spec = importlib.util.spec_from_file_location("segment_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BacktestRunner


@pytest.fixture
def build_dataset(tmp_path, monkeypatch, runner_class):
    dates = pd.to_datetime([
        "2019-12-31", "2020-01-01", "2020-06-30", "2020-12-31",
        "2021-01-01", "2021-06-30", "2021-12-31",
        "2022-01-01", "2022-06-30", "2022-12-31", "2023-01-01",
    ])
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SZ000001"]], names=["datetime", "instrument"]
    )
    features = pd.DataFrame({"factor": range(len(index))}, index=index)
    labels = pd.DataFrame({"LABEL0": range(len(index))}, index=index)
    config_text = """\
data: {}
dataset:
  label: "Ref($close, -2) / Ref($close, -1) - 1"
  segments:
    train: ["2020-01-01", "2020-12-31"]
    valid: ["2021-01-01", "2021-12-31"]
    test: ["2022-01-01", "2022-12-31"]
"""

    def build(quoted_dates=True):
        text = config_text
        if not quoted_dates:
            # YAML also supports unquoted dates (datetime.date objects).
            for year in (2020, 2021, 2022):
                for day in ("01-01", "12-31"):
                    text = text.replace(f'"{year}-{day}"', f"{year}-{day}")
        config_path = tmp_path / "backtest.yaml"
        config_path.write_text(text, encoding="utf-8")
        runner = runner_class(str(config_path))
        monkeypatch.setattr(runner, "_compute_label", lambda expression: labels.copy())
        dataset = runner._create_dataset_with_computed_factors({}, features.copy())
        return dataset, index

    return build


@pytest.fixture
def build_unit_dataset(monkeypatch, build_dataset):
    def build(quoted_dates=True):
        # No selector or filtering logic is duplicated in these stand-ins.
        modules = {name: ModuleType(name) for name in (
            "qlib", "qlib.data", "qlib.data.dataset",
            "qlib.data.dataset.handler", "qlib.data.dataset.processor",
        )}
        modules["qlib.data"].D = None
        modules["qlib.data.dataset"].DatasetH = SimpleNamespace
        modules["qlib.data.dataset.handler"].DataHandler = object
        for name in ("Fillna", "ProcessInf", "CSRankNorm", "DropnaLabel"):
            setattr(modules["qlib.data.dataset.processor"], name, object)
        with monkeypatch.context() as context:
            for name, module in modules.items():
                context.setitem(sys.modules, name, module)
            return build_dataset(quoted_dates)

    return build


def assert_segment_index(frame, segment):
    year = {"train": 2020, "valid": 2021, "test": 2022}[segment]
    expected = pd.MultiIndex.from_product(
        [pd.to_datetime([f"{year}-01-01", f"{year}-06-30", f"{year}-12-31"]),
         ["SH600000", "SZ000001"]],
        names=["datetime", "instrument"],
    )
    pd.testing.assert_index_equal(frame.index, expected)


@pytest.mark.parametrize("quoted_dates", [True, False], ids=["strings", "dates"])
@pytest.mark.parametrize("col_set", ["feature", "label", ["feature", "label"], "__all"])
def test_yaml_list_segments_are_disjoint(build_unit_dataset, quoted_dates, col_set):
    dataset, _ = build_unit_dataset(quoted_dates)
    indices = []
    for segment, selector in dataset.segments.items():
        assert isinstance(selector, list)
        result = dataset.handler.fetch(selector, col_set=col_set)
        assert_segment_index(result, segment)
        indices.append(result.index)
    for i, index in enumerate(indices):
        for other in indices[i + 1:]:
            assert index.intersection(other).empty


@pytest.mark.parametrize("selector_type", [tuple, slice], ids=["tuple", "slice"])
@pytest.mark.parametrize("segment", ["train", "valid", "test"])
def test_existing_range_selectors_still_work(build_unit_dataset, selector_type, segment):
    dataset, _ = build_unit_dataset()
    bounds = dataset.segments[segment]
    selector = slice(*bounds) if selector_type is slice else tuple(bounds)
    result = dataset.handler.fetch(selector, col_set=["feature", "label"])
    assert_segment_index(result, segment)


@pytest.mark.parametrize("col_set", ["feature", "label", ["feature", "label"], "__all"])
def test_no_selector_preserves_all_rows(build_unit_dataset, col_set):
    dataset, index = build_unit_dataset()
    result = dataset.handler.fetch(None, col_set=col_set)
    pd.testing.assert_index_equal(result.index, index)


def test_out_of_range_list_returns_empty(build_unit_dataset):
    dataset, _ = build_unit_dataset()
    result = dataset.handler.fetch(["2030-01-01", "2030-12-31"])
    assert result.empty
    assert list(result.columns) == ["factor"]


def test_filtering_does_not_mutate_source_and_preserves_squeeze(build_unit_dataset):
    dataset, index = build_unit_dataset()
    handler = dataset.handler
    before = handler.fetch(None, col_set="__all")
    result = handler.fetch(dataset.segments["train"], squeeze=True)
    assert isinstance(result, pd.Series)
    assert_segment_index(result, "train")
    result.iloc[0] = -999
    pd.testing.assert_frame_equal(handler.fetch(None, col_set="__all"), before)
    pd.testing.assert_index_equal(before.index, index)


def test_real_qlib_dataset_prepare_respects_segments(build_dataset):
    pytest.importorskip("qlib", reason="Integration test requires pyqlib")
    dataset, _ = build_dataset()
    # Matches the grouped feature/label request made during model training.
    train, valid, test = dataset.prepare(
        ["train", "valid", "test"], col_set=["feature", "label"], data_key="learn"
    )
    for frame, name in zip((train, valid, test), ("train", "valid", "test")):
        assert_segment_index(frame, name)
        assert set(frame.columns.get_level_values(0)) == {"feature", "label"}
