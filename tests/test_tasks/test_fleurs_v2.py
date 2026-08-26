"""Regression tests for the recording-level FLEURS `.v2` retrieval tasks.

FLEURS `id` identifies a *sentence*, not a recording: each sentence is read by up to
six different speakers who all share one `id`. v1 keyed the qrels on that id with a
plain dict comprehension, so every recording after the first collapsed away silently
(77,809 recordings -> 33,018 creditable ids across the 102 languages).

These tests exercise the `.v2` construction against a synthetic split that reproduces
the shapes that matter -- singleton / 2- / 3- / 6-recording groups, and two distinct
sentence ids carrying byte-identical text -- so they run offline and deterministically.
See https://github.com/embeddings-benchmark/mteb/issues/5270.
"""

from __future__ import annotations

import datasets
import pytest
from datasets import Dataset

import mteb
from mteb.tasks.retrieval.multilingual.fleurs import (
    _audio_paths,
    _build_recording_level_split,
    _recording_ids,
)

# sentence id -> (transcription, [audio filenames, one per speaker])
_SYNTHETIC = {
    1: ("one speaker read this", ["b.wav"]),
    2: ("two speakers read this", ["d.wav", "a.wav"]),
    3: ("three speakers read this", ["z.wav", "m.wav", "c.wav"]),
    4: ("six speakers read this", ["f.wav", "e.wav", "k.wav", "j.wav", "h.wav", "g.wav"]),
    5: ("a duplicated sentence", ["p.wav"]),
    6: ("a duplicated sentence", ["q.wav"]),  # same text, different sentence id
}
_N_ROWS = sum(len(paths) for _, paths in _SYNTHETIC.values())  # 14
_N_SENTENCES = len(_SYNTHETIC)  # 6


def _synthetic_split(order: list[int] | None = None) -> Dataset:
    """A FLEURS-shaped split. Audio bytes are deliberately not decodable: any code
    path that decodes audio during construction will blow up here.
    """
    ids, transcriptions, audio = [], [], []
    for sentence_id, (text, paths) in _SYNTHETIC.items():
        for path in paths:
            ids.append(sentence_id)
            transcriptions.append(text)
            audio.append({"bytes": b"not-decodable-audio", "path": path})

    # Built as a plain struct then cast, rather than encoded through the Audio
    # feature -- encoding would require `torchcodec` and would reject these bytes.
    ds = Dataset.from_dict(
        {"id": ids, "transcription": transcriptions, "audio": audio},
        features=datasets.Features(
            {
                "id": datasets.Value("int32"),
                "transcription": datasets.Value("string"),
                "audio": {
                    "bytes": datasets.Value("binary"),
                    "path": datasets.Value("string"),
                },
            }
        ),
    ).cast_column("audio", datasets.Audio(sampling_rate=16000))
    return ds.select(order) if order is not None else ds


@pytest.fixture
def built():
    return _build_recording_level_split(_synthetic_split())


# --------------------------------------------------------------------------- ids


def test_ids_are_unique_in_both_directions(built):
    audio_ds, text_ds, _, _ = built
    assert len(set(audio_ds["id"])) == len(audio_ds) == _N_ROWS
    assert len(set(text_ds["id"])) == len(text_ds) == _N_SENTENCES


def test_every_physical_recording_is_retained(built):
    """The v1 defect: rows sharing a sentence id were dropped from the label set."""
    audio_ds, _, _, _ = built
    assert len(audio_ds) == _N_ROWS
    paths = _audio_paths(audio_ds)
    expected = sorted(p for _, ps in _SYNTHETIC.values() for p in ps)
    assert sorted(paths) == expected


def test_recording_ids_are_row_order_independent():
    """Direct regression test for the v1 bug, whose surviving row depended on order."""
    forward = _synthetic_split()
    shuffled_order = [7, 0, 13, 2, 9, 4, 11, 1, 12, 5, 8, 3, 10, 6]
    shuffled = _synthetic_split(order=shuffled_order)

    fwd_audio, _, _, _ = _build_recording_level_split(forward)
    shf_audio, _, _, _ = _build_recording_level_split(shuffled)

    def path_to_id(ds):
        return dict(zip(_audio_paths(ds), ds["id"]))

    assert path_to_id(fwd_audio) == path_to_id(shf_audio)


def test_recording_id_format_and_ranking():
    """Rank is assigned by audio filename, so ids are stable and human-traceable."""
    ids = ["7", "7", "7"]
    assert _recording_ids(ids, ["z.wav", "a.wav", "m.wav"]) == ["7-2", "7-0", "7-1"]


# ------------------------------------------------------------------------- qrels


def test_t2a_qrels_are_multi_positive(built):
    """The paper scores text->speech as 'retrieving any of the speakers'."""
    _, _, t2a, _ = built
    assert len(t2a) == _N_SENTENCES
    for sentence_id, (_, paths) in _SYNTHETIC.items():
        assert len(t2a[str(sentence_id)]) == len(paths)
    assert sum(len(v) for v in t2a.values()) == _N_ROWS
    assert max(len(v) for v in t2a.values()) == 6


def test_a2t_qrels_are_single_positive(built):
    _, _, _, a2t = built
    assert len(a2t) == _N_ROWS
    # sentence ids 5 and 6 share identical text, so their recordings have 2 positives
    ambiguous = {"5-0", "6-0"}
    for recording_id, positives in a2t.items():
        assert len(positives) == (2 if recording_id in ambiguous else 1)


def test_a2t_identical_text_across_sentence_ids_is_multi_positive(built):
    """Two distinct sentence ids can carry byte-identical text (one pair in `ln_cd`);
    those documents are indistinguishable, so both must count as correct.
    """
    _, _, _, a2t = built
    assert a2t["5-0"] == {"5": 1, "6": 1}
    assert a2t["6-0"] == {"5": 1, "6": 1}


@pytest.mark.parametrize("direction", ["t2a", "a2t"])
def test_qrels_reference_only_existing_ids(built, direction):
    audio_ds, text_ds, t2a, a2t = built
    qrels, queries, corpus = (
        (t2a, text_ds, audio_ds) if direction == "t2a" else (a2t, audio_ds, text_ds)
    )
    assert set(qrels) <= set(queries["id"])
    assert {doc for positives in qrels.values() for doc in positives} <= set(corpus["id"])


@pytest.mark.parametrize("direction", ["t2a", "a2t"])
def test_every_query_has_at_least_one_positive(built, direction):
    _, _, t2a, a2t = built
    qrels = t2a if direction == "t2a" else a2t
    assert all(len(positives) > 0 for positives in qrels.values())


def test_directions_are_mirrors(built):
    """T2A queries are A2T documents and vice versa -- the same data, roles swapped."""
    audio_ds, text_ds, t2a, a2t = built
    assert set(t2a) == set(text_ds["id"])
    assert set(a2t) == set(audio_ds["id"])
    # every T2A positive is exactly one A2T query
    assert {doc for positives in t2a.values() for doc in positives} == set(a2t)
    assert sum(len(v) for v in t2a.values()) == len(a2t) == _N_ROWS


def test_construction_never_decodes_audio(built):
    """The audio bytes in the fixture are garbage; reaching this point means no
    decode happened. v1 decoded every clip just to read `id` (see issue #5270).
    """
    audio_ds, _, _, _ = built
    assert audio_ds.features["audio"].decode is True


# ---------------------------------------------------------------------- metadata


@pytest.mark.parametrize(
    "v1_name,v2_name",
    [
        ("FleursT2ARetrieval", "FleursT2ARetrieval.v2"),
        ("FleursA2TRetrieval", "FleursA2TRetrieval.v2"),
    ],
)
def test_v2_metadata(v1_name, v2_name):
    v1, v2 = mteb.get_task(v1_name).metadata, mteb.get_task(v2_name).metadata
    assert v2.adapted_from == [v1_name]
    assert v2.dataset["revision"] == v1.dataset["revision"]
    assert v2.dataset["path"] == v1.dataset["path"]
    assert v2.main_score == v1.main_score
    assert v2.category == v1.category
    assert v2.modalities == v1.modalities
    assert len(v2.eval_langs) == len(v1.eval_langs) == 102


@pytest.mark.parametrize(
    "v1_name,v2_name",
    [
        ("FleursT2ARetrieval", "FleursT2ARetrieval.v2"),
        ("FleursA2TRetrieval", "FleursA2TRetrieval.v2"),
    ],
)
def test_v1_data_is_unchanged_but_marked_superseded(v1_name, v2_name):
    """v1's construction, dataset revision and score are untouched so published
    results stay valid; only the supersession pointer is added, following the
    `BSARDRetrieval` -> `BSARDRetrieval.v2` convention.
    """
    v1 = mteb.get_task(v1_name).metadata
    assert v1.dataset["revision"] == "cadac46d4cd7d721f5cf8844a5b9f0f20e6fcde8"
    assert v1.main_score == "hit_rate_at_5"
    assert v1.adapted_from is None  # v1 is the original, not adapted from anything
    assert v1.superseded_by == v2_name


@pytest.mark.parametrize("v1_name", ["FleursT2ARetrieval", "FleursA2TRetrieval"])
def test_superseding_v1_does_not_remove_it_from_explicit_benchmarks(v1_name):
    """`get_tasks(tasks=[...])` short-circuits before `exclude_superseded`, so
    MAEB(beta) keeps referencing v1 by name.
    """
    assert mteb.get_tasks(tasks=[v1_name])[0].metadata.name == v1_name
