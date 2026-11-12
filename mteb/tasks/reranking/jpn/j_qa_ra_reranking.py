from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class JQaRAReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JQaRAReranking",
        description=(
            "JQaRA: Japanese Question Answering with Retrieval Augmentation "
            " - 検索拡張(RAG)評価のための日本語 Q&A データセット. JQaRA is an information retrieval task "
            "for questions against 100 candidate data (including one or more correct answers)."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JQaRA",
        dataset={
            "path": "mteb/JQaRAReranking",
            "revision": "63254fa512e133e6300e35b6f9175c32c6cc8455",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="map_at_1000",
        date=("2020-01-01", "2024-12-31"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=["jpn-Jpan"],
        sample_creation="found",
        prompt="Given a Japanese question, rerank passages based on their relevance for answering the question",
        bibtex_citation=r"""
@inproceedings{li-etal-2026-jmteb,
  address = {Palma de Mallorca, Spain},
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  doi = {10.63317/5ouzpv2f2f6k},
  month = may,
  pages = {7423--7434},
  publisher = {ELRA Language Resource Association},
  title = {{JMTEB} and {JMTEB}-lite: {J}apanese Massive Text Embedding Benchmark and Its Lightweight Version},
  url = {https://aclanthology.org/2026.lrec-1.588/},
  year = {2026},
}
""",
    )
