from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LASSA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LASSA2TRetrieval",
        description=(
            "Language-Queried Audio Source Separation (LASS) dataset for audio-to-text retrieval. "
            "Retrieve text descriptions/captions for audio clips using natural language queries."
            "The original dataset is based on the AudioCaps dataset."
            "The source audio has been synthesized by mixing two audio with their labelled snr ratio as indicated in the dataset."
        ),
        reference="https://dcase.community/challenge2024/task-language-queried-audio-source-separation",
        dataset={
            "path": "mteb/lass-synth-a2t",
            "revision": "8bb471ab5f4268d41fb2970af17eaaae88cc85ef",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2024-03-27", "2024-03-27"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{liu2022separate,
  author = {Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
  booktitle = {INTERSPEEH},
  title = {Separate What You Describe: Language-Queried Audio Source Separation},
  year = {2022},
}
""",
    )


class LASST2ARetrieval(AbsTaskRetrieval):
    """Text-to-audio retrieval on LASS dataset."""

    metadata = TaskMetadata(
        name="LASST2ARetrieval",
        description=(
            "Language-Queried Audio Source Separation (LASS) dataset for text-to-audio retrieval. "
            "Retrieve audio clips corresponding to natural language text descriptions/captions."
            "The original dataset is based on the AudioCaps dataset."
            "The source audio has been synthesized by mixing two audio with their labelled snr ratio as indicated in the dataset."
        ),
        reference="https://dcase.community/challenge2024/task-language-queried-audio-source-separation",
        dataset={
            "path": "mteb/lass-synth-t2a",
            "revision": "c211f7b3d3da571e9d9fda44bd2fdcc6b55f50d0",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2026-01-16", "2026-01-16"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{liu2022separate,
  author = {Liu, Xubo and Liu, Haohe and Kong, Qiuqiang and Mei, Xinhao and Zhao, Jinzheng and Huang, Qiushi and Plumbley, Mark D and Wang, Wenwu},
  booktitle = {INTERSPEEH},
  title = {Separate What You Describe: Language-Queried Audio Source Separation},
  year = {2022},
}
""",
    )
