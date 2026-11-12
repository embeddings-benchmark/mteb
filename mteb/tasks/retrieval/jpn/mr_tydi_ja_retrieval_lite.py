from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MrTyDiJaRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MrTyDiJaRetrievalLite",
        dataset={
            "path": "mteb/MrTyDiJaRetrievalLite",
            "revision": "b87e6ff25f4e32d1c97498a539ea8aad5fde3cb1",
        },
        description=(
            "Mr.TyDi is a multilingual benchmark dataset built on TyDi for document retrieval tasks. "
            "This is the lightweight Japanese version with a reduced corpus (93,382 documents) constructed using "
            "hard negatives from 5 high-performance models."
        ),
        reference="https://huggingface.co/datasets/castorini/mr-tydi",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2025-01-01"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        adapted_from=["MrTidyRetrieval"],
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

@inproceedings{mrtydi,
  address = {Punta Cana, Dominican Republic},
  author = {Zhang, Xinyu and Ma, Xueguang and Shi, Peng and Lin, Jimmy},
  booktitle = {Proceedings of the 1st Workshop on Multilingual Representation Learning},
  doi = {10.18653/v1/2021.mrl-1.12},
  editor = {Ataman, Duygu and Birch, Alexandra and Conneau, Alexis and Firat, Orhan and Ruder, Sebastian and Sahin, Gozde Gul},
  month = {nov},
  pages = {127--137},
  publisher = {Association for Computational Linguistics},
  title = {Mr. {T}y{D}i: A Multi-lingual Benchmark for Dense Retrieval},
  url = {https://aclanthology.org/2021.mrl-1.12/},
  year = {2021},
}
""",
    )
