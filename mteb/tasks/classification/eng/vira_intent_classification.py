from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class ViraIntentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ViraIntentClassification",
        description="Chatbot-delivered COVID-19 vaccine communication message preferences of young adults and public health workers in urban American communities: qualitative study",
        dataset={
            "path": "DeepPavlov/vira-intents-live",
            "revision": "a4141f65d270c89e13d269099aa9a6f188fa09f1",
        },
        reference="https://huggingface.co/datasets/ibm-research/vira-intents-live",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["val", "test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2022-07-06"),
        domains=["Medical"],
        task_subtypes=["Intent classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{weeks2022chatbot,
          title={Chatbot-delivered COVID-19 vaccine communication message preferences of young adults and public health workers in urban American communities: qualitative study},
          author={Weeks, Rose and Cooper, Lyra and Sangha, Pooja and Sedoc, Jo{\~a}o and White, Sydney and Toledo, Assaf and Gretz, Shai and Lahav, Dan and Martin, Nina and Michel, Alexandra and others},
          journal={Journal of medical Internet research},
          volume={24},
          number={7},
          pages={e38418},
          year={2022},
          publisher={JMIR Publications Toronto, Canada}
        }
        """,
    )
