from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HotpotQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HotpotQA",
        hf_hub_name="mteb/hotpotqa",
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
            " supervision for supporting facts to enable more explainable question answering systems."
        ),
        reference="https://hotpotqa.github.io/",
        type="Retrieval",
        category="s2p",
        eval_splits=["train", "dev", "test"],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        revision="ab518f4d6fcca38d87c25209f94beba119d02014",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )
