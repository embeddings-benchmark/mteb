from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class CQADupstackMathematicaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CQADupstackMathematicaRetrieval",
        description="CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        dataset={
            "path": "mteb/cqadupstack-mathematica",
            "revision": "90fceea13679c63fe563ded68f3b6f06e50061de",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Written", "Academic"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{hoogeveen2015,
author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
series = {ADCS '15},
year = {2015},
isbn = {978-1-4503-4040-3},
location = {Parramatta, NSW, Australia},
pages = {3:1--3:8},
articleno = {3},
numpages = {8},
url = {http://doi.acm.org/10.1145/2838931.2838934},
doi = {10.1145/2838931.2838934},
acmid = {2838934},
publisher = {ACM},
address = {New York, NY, USA},
}""",
    )
