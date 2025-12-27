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
@misc{jmteb_lite,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
and Kawahara, Daisuke},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
  title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
  year = {2025},
}

@article{mrtydi,
  author = {Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
  journal = {arXiv:2108.08787},
  title = {{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval},
  year = {2021},
}
""",
    )
