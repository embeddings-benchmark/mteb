from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.sts.multilingual.sts17_multilingual_visual_sts import (
    STS17MultilingualVisualSTS,
)

task_list_sts17: list[AbsTask] = [
    STS17MultilingualVisualSTS().filter_languages(
        languages=["eng"], hf_subsets=["en-en"]
    )
]


class STS17MultilingualVisualSTSEng(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="VisualSTS17Eng",
        description="STS17MultilingualVisualSTS English only.",
        reference="https://arxiv.org/abs/2402.08183/",
        tasks=task_list_sts17,
        category="i2i",
        license="not specified",
        modalities=["image"],
        annotations_creators="human-annotated",
        dialect=[""],
        eval_langs={
            "en-en": ["eng-Latn"]
        },  # rely on subsets to filter scores in TaskResults.get_score_fast().
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="VisualSTS(eng)",
        eval_splits=["test"],
        bibtex_citation=r"""
@article{xiao2024pixel,
  author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2402.08183},
  title = {Pixel Sentence Representation Learning},
  year = {2024},
}
""",
    )
