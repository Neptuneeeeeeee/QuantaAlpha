"""Regression tests for workspace data linking on supported platforms."""

import importlib.util
import os
import platform
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def experiment_module(tmp_path, monkeypatch):
    """Load the real experiment module without importing project settings."""
    conf = ModuleType("quantaalpha.core.conf")
    conf.RD_AGENT_SETTINGS = SimpleNamespace(workspace_path=tmp_path / "workspaces")
    monkeypatch.setitem(sys.modules, "quantaalpha.core.conf", conf)

    path = Path(__file__).resolve().parents[2] / "quantaalpha/core/experiment.py"
    spec = importlib.util.spec_from_file_location("_workspace_linking_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_paths(tmp_path):
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()

    (source / "daily_pv.h5").write_text("market-data", encoding="utf-8")
    (source / "metadata.json").write_text('{"ok": true}', encoding="utf-8")
    return source, workspace


def assert_linked_contents(source, workspace):
    assert sorted(p.name for p in workspace.iterdir()) == ["daily_pv.h5", "metadata.json"]
    for src in source.iterdir():
        linked = workspace / src.name
        assert linked.read_bytes() == src.read_bytes()


@pytest.mark.skipif(platform.system() != "Darwin", reason="Requires a real macOS filesystem")
def test_real_darwin_creates_symbolic_links(experiment_module, tmp_path):
    source, workspace = make_paths(tmp_path)

    experiment_module.FBWorkspace.link_all_files_in_folder_to_workspace(source, workspace)

    assert_linked_contents(source, workspace)
    for src in source.iterdir():
        linked = workspace / src.name
        assert linked.is_symlink()
        assert linked.resolve() == src.resolve()


@pytest.mark.parametrize("system_name", ["Linux", "Darwin"])
def test_unix_platforms_use_symbolic_links(
    experiment_module, tmp_path, monkeypatch, system_name
):
    source, workspace = make_paths(tmp_path)
    monkeypatch.setattr(experiment_module.platform, "system", lambda: system_name)

    experiment_module.FBWorkspace.link_all_files_in_folder_to_workspace(source, workspace)

    assert_linked_contents(source, workspace)
    assert all((workspace / src.name).is_symlink() for src in source.iterdir())


def test_windows_uses_hard_links(experiment_module, tmp_path, monkeypatch):
    source, workspace = make_paths(tmp_path)
    monkeypatch.setattr(experiment_module.platform, "system", lambda: "Windows")

    experiment_module.FBWorkspace.link_all_files_in_folder_to_workspace(source, workspace)

    assert_linked_contents(source, workspace)
    for src in source.iterdir():
        linked = workspace / src.name
        assert not linked.is_symlink()
        assert os.stat(linked).st_ino == os.stat(src).st_ino


@pytest.mark.skipif(platform.system() != "Darwin", reason="Requires a real macOS filesystem")
def test_real_darwin_replaces_existing_destination(experiment_module, tmp_path):
    source, workspace = make_paths(tmp_path)
    stale = workspace / "daily_pv.h5"
    stale.write_text("stale", encoding="utf-8")

    experiment_module.FBWorkspace.link_all_files_in_folder_to_workspace(source, workspace)

    assert stale.is_symlink()
    assert stale.read_text(encoding="utf-8") == "market-data"
