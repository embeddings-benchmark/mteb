from __future__ import annotations

import time

import pytest

from mteb.timing import TimingStack


def test_timing_stack():
    timer1 = TimingStack()
    assert timer1.phases == []
    assert timer1._start_time is None
    assert timer1.plot() == "No timing phases recorded."

    with timer1("test_phase", split="test", subset="default"):
        time.sleep(0.01)

    assert timer1._start_time is not None
    assert len(timer1.phases) == 1
    phase = timer1.phases[0]
    assert phase["name"] == "test_phase"
    assert phase["split"] == "test"
    assert phase["subset"] == "default"
    assert phase["start"] == pytest.approx(0.0, abs=1e-3)
    assert phase["end"] > 0.0

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
    assert len(timer2.phases) == 4

    assert timer2.phases[0]["name"] == "Load Data"
    assert timer2.phases[0]["start"] == 0.0
    assert timer2.phases[0]["end"] == 2.5
    assert timer2.phases[0]["split"] == "test"
    assert timer2.phases[0]["subset"] == "en"

    assert timer2.phases[1]["name"] == "Encode"
    assert timer2.phases[1]["start"] == 2.5
    assert timer2.phases[1]["end"] == 6.0

    assert timer2.phases[2]["name"] == "Scoring"
    assert timer2.phases[2]["start"] == 6.0
    assert timer2.phases[2]["end"] == 8.0
    assert not timer2.phases[2]["split"]
    assert not timer2.phases[2]["subset"]

    assert timer2.phases[3]["name"] == "Evaluate"
    assert timer2.phases[3]["start"] == 8.0
    assert timer2.phases[3]["end"] == 10.0
    assert timer2.phases[3]["split"] == "validation"
    assert not timer2.phases[3]["subset"]

    plot_output = timer2.plot()
    assert "Load Data (test, en)" in plot_output
    assert "Encode (test, en)" in plot_output
    assert "Scoring" in plot_output
    assert "Scoring (" not in plot_output
    assert "Evaluate (validation)" in plot_output
    assert "2.5s" in plot_output
    assert "3.5s" in plot_output
    assert "2.0s" in plot_output
