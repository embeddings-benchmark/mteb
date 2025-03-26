from __future__ import annotations

from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.Image.VisualSTS import STSBenchmarkMultilingualVisualSTS

task_list_stsb: list[AbsTask] = [
    STSBenchmarkMultilingualVisualSTS().filter_languages(
        languages=["eng"], hf_subsets=["en"]
    )
]


class STSBenchmarkMultilingualVisualSTSEng(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="VisualSTS-b-Eng",
        description="STSBenchmarkMultilingualVisualSTS English only.",
        reference="https://arxiv.org/abs/2402.08183/",
        tasks=task_list_stsb,
        category="i2i",
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[""],
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="VisualSTS(eng)",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
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
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )
