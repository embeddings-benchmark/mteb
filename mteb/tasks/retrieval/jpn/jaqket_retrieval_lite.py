from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JaqketRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaqketRetrievalLite",
        dataset={
            "path": "mteb/JaqketRetrievalLite",
            "revision": "860965fbb6526dd8edff12627dacf07c8f5a54f3",
        },
        description=(
            "JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset created based on quiz questions. "
            "This is the lightweight version with a reduced corpus (65,802 documents) constructed using "
            "hard negatives from 5 high-performance models."
        ),
        reference="https://github.com/kumapo/JAQKET-dataset",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2023-10-09", "2025-01-01"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        adapted_from=["JaqketRetrieval"],
        bibtex_citation=r"""
@misc{jmteb_lite,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
and Kawahara, Daisuke},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
  title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
  year = {2025},
}

@inproceedings{Kurihara_nlp2020,
  author = {鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也},
  booktitle = {言語処理学会第26回年次大会},
  note = {in Japanese},
  title = {JAQKET: クイズを題材にした日本語 QA データセットの構築},
  url = {https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf},
  year = {2020},
}
""",
    )
