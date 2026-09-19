from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SpokenSQuADT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpokenSQuADT2ARetrieval",
        description="Text-to-audio retrieval task based on SpokenSQuAD dataset. Given a text question, retrieve relevant audio segments that contain the answer. Questions are derived from SQuAD reading comprehension dataset with corresponding spoken passages.",
        reference="https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD",
        dataset={
            "path": "mteb/spoken-squad-t2a",
            "revision": "b311394fa5ab5e003012304799d0d724d64303b3",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2018-03-01", "2018-03-01"),
        domains=["Academic", "Encyclopaedic", "Non-fiction"],
        task_subtypes=["Question Answering Retrieval", "Reading Comprehension"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lee2018spoken,
  author = {Lee, Chia-Hsuan and Wu, Szu-Lin and Liu, Chi-Liang and Lee, Hung-yi},
  journal = {Proc. Interspeech 2018},
  pages = {3459--3463},
  title = {Spoken SQuAD: A Study of Mitigating the Impact of Speech Recognition Errors on Listening Comprehension},
  year = {2018},
}
""",
    )
