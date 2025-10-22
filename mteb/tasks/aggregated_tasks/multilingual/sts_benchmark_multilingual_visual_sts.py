from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.sts.multilingual.sts_benchmark_multilingual_visual_sts import (
    STSBenchmarkMultilingualVisualSTS,
)

task_list_multi: list[AbsTask] = [
    STSBenchmarkMultilingualVisualSTS().filter_languages(
        languages=[
            "deu",
            "spa",
            "fra",
            "ita",
            "nld",
            "pol",
            "por",
            "rus",
            "cmn",
        ],
        hf_subsets=[
            "de",
            "es",
            "fr",
            "it",
            "nl",
            "pl",
            "pt",
            "ru",
            "zh",
        ],
    )
]


class STSBenchmarkMultilingualVisualSTSMultilingual(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="VisualSTS-b-Multilingual",
        description="STSBenchmarkMultilingualVisualSTS multilingual.",
        reference="https://arxiv.org/abs/2402.08183/",
        tasks=task_list_multi,
        category="i2i",
        license="not specified",
        modalities=["image"],
        annotations_creators="human-annotated",
        dialect=[""],
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="VisualSTS(multi)",
        eval_splits=["test"],
        eval_langs=[
            "deu-Latn",
            "spa-Latn",
            "fra-Latn",
            "ita-Latn",
            "nld-Latn",
            "pol-Latn",
            "por-Latn",
            "rus-Cyrl",
            "cmn-Hans",
        ],
        bibtex_citation=r"""
@article{xiao2024pixel,
  author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2402.08183},
  title = {Pixel Sentence Representation Learning},
  year = {2024},
}
""",
    )
