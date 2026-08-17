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
            "revision": "466e0ea0ed8f50bad9c240f3bfc8426c08430aa2",
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
@dataset{bongjun_kim_2019_2596911,
  author = {Bongjun Kim and
Mark Cartwright and
Fatemeh Pishdadian and
Bryan Pardo},
  doi = {10.5281/zenodo.2596911},
  month = mar,
  publisher = {Zenodo},
  title = {VimSketch Dataset},
  url = {https://doi.org/10.5281/zenodo.2596911},
  version = {1.0},
  year = {2019},
}
""",
        prompt={
            "query": "Retrieve the original sound that this vocal imitation mimics."
        },
    )
