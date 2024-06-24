from __future__ import annotations

from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata


class VoyageMMarcoReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="VoyageMMarcoReranking",
        description="a hard-negative augmented version of the Japanese MMARCO dataset as used in Voyage AI Evaluation Suite",
        reference="https://arxiv.org/abs/2312.16144",
        dataset={
            "path": "bclavie/mmarco-japanese-hard-negatives",
            "revision": "e25c91bc31859606507a968559ab1de0f472d007",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="map",
        date=("2016-12-01", "2023-12-23"),
        form=["written"],
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Scientific Reranking"],
        license="CC BY 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{clavié2023jacolbert,
      title={JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings for Retrieval: Early Technical Report}, 
      author={Benjamin Clavié},
      year={2023},
      eprint={2312.16144},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}""",
        n_samples=None,
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column(
            "positives", "positive"
        ).rename_column("negatives", "negative")
        # 391,061 dataset size
        self.dataset["test"] = self.dataset.pop("train").train_test_split(
            test_size=2048, seed=self.seed
        )["test"]
