"""Tests for run.py.

Unit tests run everywhere, including CI. The integration test needs the model
files in ./model, so it is skipped automatically when they are absent (as in CI)
and runs on the local VM.
"""

import json

import pytest

import run


def test_build_messages_wraps_prompt_as_user_message():
    assert run.build_messages("Hello") == [{"role": "user", "content": "Hello"}]


def test_tokens_per_second():
    assert run.tokens_per_second(50, 2.0) == 25.0


@pytest.mark.parametrize("elapsed", [0, -1.5])
def test_tokens_per_second_rejects_non_positive_time(elapsed):
    with pytest.raises(ValueError):
        run.tokens_per_second(10, elapsed)


def test_peak_memory_is_positive():
    assert run.peak_memory_mib() > 0


def test_save_result_round_trip(tmp_path):
    path = tmp_path / "results" / "out.json"
    record = {"prompt": "test", "output": "a poem"}
    run.save_result(path, record)
    assert json.loads(path.read_text(encoding="utf-8")) == record


@pytest.mark.skipif(not run.MODEL_DIR.exists(), reason="model files not present (CI)")
def test_generate_produces_text():
    text, count, elapsed = run.generate(run.MODEL_DIR, run.PROMPT, max_new_tokens=20)
    assert text.strip()
    assert count > 0
    assert elapsed > 0
