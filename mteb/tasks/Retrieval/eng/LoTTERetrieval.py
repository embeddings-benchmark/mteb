from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

DOMAINS = ["writing", "recreation", "science", "technology", "lifestyle"]
DOMAINS_TYPES = ["search", "forum"]
HF_SUBSETS = [f"{d}_{t}" for d in DOMAINS for t in DOMAINS_TYPES]


class LoTTERetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LoTTE",
        dataset={
            "path": "mteb/LoTTE",
            "revision": "a887e9f8e3d24427fdff98c8bba3198a6853a929",
        },
        description=(
            "LoTTE (Long-Tail Topic-stratified Evaluation for IR) is designed to evaluate retrieval models "
            "on underrepresented, long-tail topics. Unlike MSMARCO or BEIR, LoTTE features domain-specific queries and "
            "passages from StackExchange (covering writing, recreation, science, technology, and lifestyle), providing "
            "a challenging out-of-domain generalization benchmark."
        ),
        type="Retrieval",
        modalities=["text"],
        category="s2s",
        reference="https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md",
        eval_splits=["test", "dev"],
        eval_langs={domain: ["eng-Latn"] for domain in HF_SUBSETS},
        main_score="precision_at_5",
        date=("2021-12-02", "2022-06-10"),
        domains=["Academic", "Web", "Social"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{santhanam-etal-2022-colbertv2,
            title = "{C}ol{BERT}v2: Effective and Efficient Retrieval via Lightweight Late Interaction",
            author = "Santhanam, Keshav  and
              Khattab, Omar  and
              Saad-Falcon, Jon  and
              Potts, Christopher  and
              Zaharia, Matei",
            editor = "Carpuat, Marine  and
              de Marneffe, Marie-Catherine  and
              Meza Ruiz, Ivan Vladimir",
            booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
            month = jul,
            year = "2022",
            address = "Seattle, United States",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.naacl-main.272/",
            doi = "10.18653/v1/2022.naacl-main.272",
            pages = "3715--3734",
        }""",
    )
