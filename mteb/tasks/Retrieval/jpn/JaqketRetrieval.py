from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class JaqketRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaqketRetrieval",
        dataset={
            "path": "mteb/jaqket",
            "revision": "3a5b92dad489a61e664c05ed2175bc9220230199",
        },
        description="JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset that is created based on quiz questions.",
        reference="https://github.com/kumapo/JAQKET-dataset",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2023-10-09", "2023-10-09"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@InProceedings{Kurihara_nlp2020,
author =  "鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也",
title =   "JAQKET: クイズを題材にした日本語 QA データセットの構築",
booktitle =   "言語処理学会第26回年次大会",
year =    "2020",
url = "https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf"
note= "in Japanese"
}""",
    )
