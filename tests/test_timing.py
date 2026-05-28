from __future__ import annotations

import time

import pytest

from mteb.timing import TimingStack


def test_timing_stack():
    timer1 = TimingStack()
    assert timer1.phases == []
    assert timer1._start_time is None
    assert timer1.plot() == "No timing phases recorded."

    with timer1("Load Data", split="test", subset="en"):
        time.sleep(0.01)

    with timer1("Encode", split="test", subset="en"):
        time.sleep(0.01)

    assert timer1._start_time is not None
    assert len(timer1.phases) == 2

    phases_no_time = [
        {k: v for k, v in phase.items() if k not in {"start", "end"}}
        for phase in timer1.phases
    ]
    assert phases_no_time == [
        {"name": "Load Data", "split": "test", "subset": "en"},
        {"name": "Encode", "split": "test", "subset": "en"},
    ]

    assert timer1.phases[0]["start"] == pytest.approx(0.0, abs=1e-3)
    assert timer1.phases[0]["end"] > 0.0
    assert timer1.phases[1]["start"] == pytest.approx(timer1.phases[0]["end"], abs=1e-3)
    assert timer1.phases[1]["end"] > timer1.phases[1]["start"]

    timer2 = TimingStack()
    timer2.add_phase(
        name="Load Data", start=1000.0, end=1002.5, split="test", subset="en"
    )
    timer2.add_phase(name="Encode", start=1002.5, end=1006.0, split="test", subset="en")
    timer2.add_phase(name="Scoring", start=1006.0, end=1008.0, split="", subset="")
    timer2.add_phase(
        name="Evaluate", start=1008.0, end=1010.0, split="validation", subset=""
    )

    assert timer2._start_time == 1000.0
    assert timer2.phases == [
        {
            "name": "Load Data",
            "start": 0.0,
            "end": 2.5,
            "split": "test",
            "subset": "en",
        },
        {
            "name": "Encode",
            "start": 2.5,
            "end": 6.0,
            "split": "test",
            "subset": "en",
        },
        {"name": "Scoring", "start": 6.0, "end": 8.0, "split": "", "subset": ""},
        {
            "name": "Evaluate",
            "start": 8.0,
            "end": 10.0,
            "split": "validation",
            "subset": "",
        },
    ]

    plot_output = timer2.plot()
    assert "Load Data (test, en)" in plot_output
    assert "Encode (test, en)" in plot_output
    assert "Scoring" in plot_output
    assert "Scoring (" not in plot_output
    assert "Evaluate (validation)" in plot_output
    assert "2.5s" in plot_output
    assert "3.5s" in plot_output
    assert "2.0s" in plot_output
