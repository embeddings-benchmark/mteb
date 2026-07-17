from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VimSketchA2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VimSketchA2ARetrieval",
        description=(
            "Query-by-vocal-imitation retrieval on the VimSketch dataset. Queries are "
            "human vocal imitations of a sound and the corpus contains the 542 "
            "reference sounds (environmental sounds, instruments and sound effects) "
            "they imitate; the goal is to retrieve the imitated reference sound. VimSketch merges the Vocal Imitation Set and "
            "VocalSketch and provides one reference recording per sound class. "
            "Queries are downsampled to at most 4 imitations per reference "
            "(2,168 queries)."
        ),
        reference="https://zenodo.org/record/2596911",
        dataset={
            "path": "dukesun99/VimSketch",
            "revision": "7ce441ca24859315fb78c3a987d5c62d0b642cae",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2015-01-01", "2019-03-18"),
        domains=["AudioScene", "Spoken"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{cartwright2015vocalsketch,
  author = {Cartwright, Mark and Pardo, Bryan},
  booktitle = {Proceedings of the 33rd Annual ACM Conference on Human Factors in Computing Systems},
  pages = {43--46},
  title = {VocalSketch: Vocally Imitating Audio Concepts},
  year = {2015},
}

@inproceedings{kim2018vocal,
  author = {Kim, Bongjun and Ghei, Madhav and Pardo, Bryan and Duan, Zhiyao},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018)},
  pages = {148--152},
  title = {Vocal Imitation Set: a dataset of vocally imitated sound events using the AudioSet ontology},
  year = {2018},
}
""",
        prompt={
            "query": "Retrieve the original sound that this vocal imitation mimics."
        },
    )
