from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class SpokenSQuADT2ARetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="SpokenSQuADT2ARetrieval",
        description="Text-to-audio retrieval task based on SpokenSQuAD dataset. Given a text question, retrieve relevant audio segments that contain the answer. Questions are derived from SQuAD reading comprehension dataset with corresponding spoken passages.",
        reference="https://github.com/chiuwy/Spoken-SQuAD",
        dataset={
            "path": "arteemg/spoken-squad-t2a",  
            "revision": "main",  
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2018-03-01", "2018-03-01"), 
        domains=["Academic", "Encyclopaedic", "Non-fiction"],
        task_subtypes=["Question Answering Retrieval", "Reading Comprehension"],
        license="cc-by-sa-4.0",  
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{li2018spokensquad,
  title={Spoken SQuAD: A Study of Mitigating the Impact of Speech Recognition Errors on Listening Comprehension},
  author={Li, Chia-Hsuan and Ma, Szu-Lin and Zhang, Hsin-Wei and Lee, Hung-yi and Lee, Lin-shan},
  booktitle={Interspeech},
  pages={3459--3463},
  year={2018}
}
""",
    )


