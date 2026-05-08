from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.sts.multilingual.sts17_multilingual_visual_sts import (
    STS17MultilingualVisualSTS,
)

task_list_sts17_multi = [
    STS17MultilingualVisualSTS().filter_languages(
        languages=["ara", "eng", "spa", "kor"],
        hf_subsets=[
            "ko-ko",
            "ar-ar",
            "en-ar",
            "en-de",
            "en-tr",
            "es-en",
            "es-es",
            "fr-en",
            "it-en",
            "nl-en",
        ],
    )
]


class STS17MultilingualVisualSTSMultilingual(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="VisualSTS17Multilingual",
        description="STS17MultilingualVisualSTS multilingual.",
        reference="https://arxiv.org/abs/2402.08183/",
        tasks=task_list_sts17_multi,
        category="i2i",
        modalities=["image"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[""],
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="VisualSTS(multi)",
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
