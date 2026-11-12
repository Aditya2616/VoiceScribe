"""The README's numbers, commands and paths are generated from, or checked against, results and code."""

import importlib.util
import sys

import pytest
from conftest import ROOT


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_numbers", ROOT / "scripts" / "check_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_numbers"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_readme_is_in_sync_with_results_and_code(check, capsys):
    rc = check.main(["--check"])
    assert rc == 0, capsys.readouterr().out


def test_a_changed_number_is_caught(check, tmp_path, monkeypatch, capsys):
    text = (ROOT / "README.md").read_text(encoding="utf8")
    assert "<!--n:n_dev-->" in text
    edited = text.replace("<!--n:n_dev-->", "<!--n:n_dev-->9", 1)
    fake = tmp_path / "README.md"
    fake.write_text(edited, encoding="utf8")
    monkeypatch.setattr(check, "README", fake)
    assert check.main(["--check"]) == 1
    assert "out of sync" in capsys.readouterr().out


def test_a_made_up_command_flag_or_path_is_caught(check):
    text = "Run `voicescribe nonsense` or `voicescribe run --nope x` or see `results/not_there.json`."
    problems = check.check_references(text)
    assert any("nonsense" in p for p in problems)
    assert any("--nope" in p for p in problems)
    assert any("not_there" in p for p in problems)


def test_write_regenerates_the_same_text(check):
    text = (ROOT / "README.md").read_text(encoding="utf8").replace("\r\n", "\n")
    assert check.render(text, check.compute()) == text
