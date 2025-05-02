from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NorwegianParliamentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NorwegianParliamentClassification",
        description="Norwegian parliament speeches annotated for sentiment",
        reference="https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
        dataset={
            "path": "NbAiLab/norwegian_parliament",
            "revision": "f7393532774c66312378d30b197610b43d751972",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["nob-Latn"],
        # assumed to be bokmål
        main_score="accuracy",
        date=("1999-01-01", "2016-01-01"),  # based on dates within the dataset
        domains=["Government", "Spoken"],
        task_subtypes=["Political classification"],
        license="cc-by-4.0",
        annotations_creators="derived",  # based on the speaker affiliation
        dialect=[],  # unknown
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kummervold-etal-2021-operationalizing,
  abstract = {In this work, we show the process of building a large-scale training set from digital and digitized collections at a national library. The resulting Bidirectional Encoder Representations from Transformers (BERT)-based language model for Norwegian outperforms multilingual BERT (mBERT) models in several token and sequence classification tasks for both Norwegian Bokm{\aa}l and Norwegian Nynorsk. Our model also improves the mBERT performance for other languages present in the corpus such as English, Swedish, and Danish. For languages not included in the corpus, the weights degrade moderately while keeping strong multilingual properties. Therefore, we show that building high-quality models within a memory institution using somewhat noisy optical character recognition (OCR) content is feasible, and we hope to pave the way for other memory institutions to follow.},
  address = {Reykjavik, Iceland (Online)},
  author = {Kummervold, Per E  and
De la Rosa, Javier  and
Wetjen, Freddy  and
Brygfjeld, Svein Arne},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {20--29},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {Operationalizing a National Digital Library: The Case for a {N}orwegian Transformer Model},
  url = {https://aclanthology.org/2021.nodalida-main.3},
  year = {2021},
}
""",
        prompt="Classify parliament speeches in Norwegian based on political affiliation",
    )
