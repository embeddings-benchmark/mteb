from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JaCWIRRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaCWIRRetrievalLite",
        dataset={
            "path": "mteb/JaCWIRRetrievalLite",
            "revision": "79472b360242cf2692e24a6d9999ef50d350d672",
        },
        description=(
            "JaCWIR (Japanese Casual Web IR) is a dataset consisting of questions and webpage meta descriptions "
            "collected from Hatena Bookmark. This is the lightweight version with a reduced corpus "
            "(302,638 documents) constructed using hard negatives from 5 high-performance models."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JaCWIR",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-01-01"),
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["JaCWIRRetrieval"],
        bibtex_citation=r"""
@misc{jmteb_lite,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
and Kawahara, Daisuke},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
  title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
  year = {2025},
}

@misc{yuichi-tateno-2024-jacwir,
  author = {Yuichi Tateno},
  title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
  url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
}
""",
    )
