from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NarrativeQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="NarrativeQARetrieval",
        dataset={
            "path": "mteb/NarrativeQARetrieval",
            "revision": "64ae1960701b9830697269b57d7420dc5c41eb58",
        },
        reference="https://metatext.io/datasets/narrativeqa",
        description=(
            "NarrativeQA is a dataset for the task of question answering on long narratives. It consists of "
            "realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. "
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Fiction", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{kocisky-etal-2018-narrativeqa,
  address = {Cambridge, MA},
  author = {Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
Schwarz, Jonathan  and
Blunsom, Phil  and
Dyer, Chris  and
Hermann, Karl Moritz  and
Melis, G{\'a}bor  and
Grefenstette, Edward},
  doi = {10.1162/tacl_a_00023},
  editor = {Lee, Lillian  and
Johnson, Mark  and
Toutanova, Kristina  and
Roark, Brian},
  journal = {Transactions of the Association for Computational Linguistics},
  pages = {317--328},
  publisher = {MIT Press},
  title = {The {N}arrative{QA} Reading Comprehension Challenge},
  url = {https://aclanthology.org/Q18-1023},
  volume = {6},
  year = {2018},
}
""",
    )
