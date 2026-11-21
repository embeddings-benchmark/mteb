from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LEMBNarrativeQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBNarrativeQARetrieval",
        dataset={
            "path": "mteb/LEMBNarrativeQARetrieval",
            "revision": "abbd273d020ec661d75fb12d4fd8ff5e9dd50793",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("narrativeqa subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1000-01-01", "2017-12-31"),
        domains=["Fiction", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
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
