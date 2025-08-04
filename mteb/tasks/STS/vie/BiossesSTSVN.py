from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class BiossesSTSVN(AbsTaskSTS):
    metadata = TaskMetadata(
        name="BIOSSES-VN",
        dataset={
            "path": "GreenNode/biosses-sts-vn",
            "revision": "1dae4a6df91c0852680cd4ab48c8c1d8a9ed49b2",
        },
        description="""A translated dataset from Biomedical Semantic Similarity Estimation.
            The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system:
            - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation.
            - Applies advanced embedding models to filter the translations.
            - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.""",
        reference="https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="cosine_spearman",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Medical"],
        task_subtypes=[],
        bibtex_citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["BIOSSES"],
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
