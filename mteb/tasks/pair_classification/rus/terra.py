from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType

_terra_metadata = dict(
    dataset={
        "path": "mteb/TERRa",
        "revision": "48312c334eb916d88481dc4a41bd0d8b2fa0d576",
    },
    reference="https://aclanthology.org/2020.emnlp-main.381/",
    type="PairClassification",
    category="t2t",
    modalities=["text"],
    eval_splits=["dev"],
    eval_langs=["rus-Cyrl"],
    main_score="max_ap",
    date=("2000-01-01", "2018-01-01"),
    domains=["News", "Web", "Written"],
    task_subtypes=[],
    license="mit",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@inproceedings{shavrina2020russiansuperglue,
  address = {Online},
  author = {Shavrina, Tatiana and Fenogenova, Alena and Anton, Emelyanov and Shevelev, Denis and Artemova, Ekaterina and Malykh, Valentin and Mikhailov, Vladislav and Tikhonova, Maria and Chertok, Andrey and Evlampiev, Andrey},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  doi = {10.18653/v1/2020.emnlp-main.381},
  editor = {Webber, Bonnie and Cohn, Trevor and He, Yulan and Liu, Yang},
  month = {nov},
  pages = {4717--4726},
  publisher = {Association for Computational Linguistics},
  title = {{R}ussian{SuperGLUE}: A {R}ussian Language Understanding Evaluation Benchmark},
  url = {https://aclanthology.org/2020.emnlp-main.381/},
  year = {2020},
}
""",
)


class TERRa(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TERRa",
        description="Textual Entailment Recognition for Russian. This task requires to recognize, given two text fragments, "
        "whether the meaning of one text is entailed (can be inferred) from the other text.",
        prompt="Given a premise, retrieve a hypothesis that is entailed by the premise",
        **_terra_metadata,
    )


class TERRaV2(AbsTaskPairClassification):
    input1_prompt_type = PromptType.document
    input2_prompt_type = PromptType.query

    metadata = TaskMetadata(
        name="TERRa.V2",
        description="Textual Entailment Recognition for Russian. This task requires to recognize, given two text fragments, "
        "whether the meaning of one text is entailed (can be inferred) from the other text."
        " Version 2 uses different prompt types for the two inputs.",
        adapted_from=["TERRa"],
        prompt={
            PromptType.query.value: "Given a premise, retrieve a hypothesis that is entailed by the premise"
        },
        **_terra_metadata,
    )
