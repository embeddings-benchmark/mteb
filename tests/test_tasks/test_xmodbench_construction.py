from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.data.xmodbench.create_data import (
    EXCLUDED_SOURCE_ROWS,
    EXCLUDED_VIDEO_PATHS,
    _exclusion_for_source_row,
    clean_question,
    convert_source_row,
)


def _source_row(
    *,
    condition_modality: str,
    condition_input: str,
    candidate_modality: str,
    question: str = "Which candidate matches? Answer with A, B, C, or D",
) -> dict:
    return {
        "index": 0,
        "subtask": "01_perception/natures",
        "question": question,
        "conditions": {
            "modality": condition_modality,
            "input": condition_input,
        },
        "options": {
            letter: {
                "modality": candidate_modality,
                "input": (
                    f"candidate {letter}"
                    if candidate_modality == "Text"
                    else f"Data/candidates/{letter}.{candidate_modality.casefold()}"
                ),
            }
            for letter in ("A", "B", "C", "D")
        },
        "correct_answer": "B",
        "category": "01_perception/natures",
    }


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which clip matches? Choose A, B, C, or D.", "Which clip matches?"),
        ("Which clip matches? Answer with A, B, C, or D", "Which clip matches?"),
        (
            "Which clip matches? Answer the question with A, B, C, or D.",
            "Which clip matches?",
        ),
        (
            "Which direction matches? (0° = front, +90° = left)",
            "Which direction matches? (0° = front, +90° = left)",
        ),
    ],
)
def test_clean_question(question: str, expected: str) -> None:
    assert clean_question(question) == expected


def test_convert_media_condition_to_image_candidates() -> None:
    row = _source_row(
        condition_modality="Audio",
        condition_input="Data/audio/example.wav",
        candidate_modality="Image",
    )

    direction, query, corpus, qrel, top_ranked, metadata = convert_source_row(
        row, "a2v", 0
    )

    assert direction == "at2i"
    assert query == {
        "id": "xmodbench_lite_a2v_0000",
        "text": "Which candidate matches?",
        "audio": "Data/audio/example.wav",
    }
    assert len(corpus) == 4
    assert corpus[1] == {
        "id": "xmodbench_lite_a2v_0000_B",
        "image": "Data/candidates/B.image",
    }
    assert qrel == {
        "query-id": "xmodbench_lite_a2v_0000",
        "corpus-id": "xmodbench_lite_a2v_0000_B",
        "score": 1,
    }
    assert top_ranked["corpus-ids"] == [item["id"] for item in corpus]
    assert metadata["source_config"] == "a2v"
    assert metadata["family"] == "perception"


def test_convert_text_condition_preserves_source_prompt_layout() -> None:
    row = _source_row(
        condition_modality="Text",
        condition_input="Thunder",
        candidate_modality="Audio",
        question=(
            "Based on this description: 'thunder', which clip matches? "
            "Choose A, B, C, or D."
        ),
    )

    direction, query, *_ = convert_source_row(row, "t2a", 0)

    assert direction == "t2a"
    assert query["text"] == (
        "Context: Thunder\n\nBased on this description: 'thunder', which clip matches?"
    )
    assert query["text"].casefold().count("thunder") == 2


def test_convert_rejects_mixed_candidate_modalities() -> None:
    row = _source_row(
        condition_modality="Audio",
        condition_input="Data/audio/example.wav",
        candidate_modality="Image",
    )
    invalid_row = deepcopy(row)
    invalid_row["options"]["D"]["modality"] = "Video"

    with pytest.raises(ValueError, match="mixes candidate modalities"):
        convert_source_row(invalid_row, "a2v", 0)


def test_malformed_video_exclusion_is_explicit_and_auditable() -> None:
    row = _source_row(
        condition_modality="Audio",
        condition_input="Data/audio/example.wav",
        candidate_modality="Video",
    )
    row["index"] = 104
    row["options"]["B"]["input"] = "Data/ExtremCountAV/hPuylJBmk_8.mp4"

    exclusion = _exclusion_for_source_row(row, "a2v", 104)

    assert exclusion is not None
    assert exclusion["query_id"] == "xmodbench_lite_a2v_0104"
    assert exclusion["direction"] == "at2v"
    assert exclusion["invalid_media_uses"] == ["option:B"]
    assert exclusion["affects_query_or_correct_answer"] is True
    assert len(EXCLUDED_SOURCE_ROWS) == 19
    assert len(EXCLUDED_VIDEO_PATHS) == 5


def test_undeclared_malformed_video_reference_is_rejected() -> None:
    row = _source_row(
        condition_modality="Audio",
        condition_input="Data/audio/example.wav",
        candidate_modality="Video",
    )
    row["options"]["A"]["input"] = "Data/ExtremCountAV/hPuylJBmk_8.mp4"

    with pytest.raises(ValueError, match="Undeclared malformed media reference"):
        _exclusion_for_source_row(row, "a2v", 105)
