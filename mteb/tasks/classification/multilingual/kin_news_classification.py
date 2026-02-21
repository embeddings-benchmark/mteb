from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class KinNewsClassification(MultilingualTask, AbsTaskClassification):
    """
    KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi.
    Each sample is a news article from Rwanda and Burundi news websites and newspapers, 
    classified into one of 14 possible classes.
    """

    metadata = TaskMetadata(
        name="KinNewsClassification",
        description=(
            "Kinyarwanda and Kirundi news classification datasets (KINNEWS and KIRNEWS, respectively), "
            "which were both collected from Rwanda and Burundi news websites and newspapers, "
            "for low-resource monolingual and cross-lingual multiclass classification tasks. "
            "Each news article can be classified into one of 14 possible classes: politics, sport, "
            "economy, health, entertainment, history, technology, culture, religion, environment, "
            "education, relationship."
        ),
        reference="https://arxiv.org/abs/2010.12174",
        dataset={
            "path": "andreniyongabo/kinnews_kirnews",
            "revision": "main",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        # Use HF builder config names as keys so MultiSubsetLoader loads correctly
        eval_langs={
            "kinnews_cleaned": ["kin-Latn"],  # Kinyarwanda
            "kirnews_cleaned": ["run-Latn"],  # Kirundi (ISO-639-3: run)
        },
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{niyongabo2020kinnews,
  title={KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi},
  author={Niyongabo, Rubungo Andre and Qu, Hong and Kreutzer, Julia and Huang, Li},
  journal={arXiv preprint arXiv:2010.12174},
  year={2020}
}
""",
    )

    def dataset_transform(self) -> None:
        """
        Transform the dataset to MTEB expected format:
        
        * column **text**: concatenation of title and content
        * column **label**: int (0-13 for the 14 news topics)
        """
        for lang in self.dataset:
            for split in self.dataset[lang]:
                ds = self.dataset[lang][split]
                
                def transform_example(example):
                    # Concatenate title and content for the text field
                    text = f"{example['title']} {example['content']}"
                    
                    return {
                        "text": text,
                        "label": example["label"]
                    }
                
                ds = ds.map(
                    transform_example,
                    remove_columns=ds.column_names,
                    desc=f"{lang}/{split}",
                )
                
                self.dataset[lang][split] = ds 