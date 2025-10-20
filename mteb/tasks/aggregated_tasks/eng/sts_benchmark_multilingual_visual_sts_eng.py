from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.sts.multilingual.sts_benchmark_multilingual_visual_sts import (
    STSBenchmarkMultilingualVisualSTS,
)

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
        modalities=["image"],
        annotations_creators="human-annotated",
        dialect=[""],
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="VisualSTS(eng)",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        bibtex_citation=r"""
@article{xiao2024pixel,
  author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2402.08183},
  title = {Pixel Sentence Representation Learning},
  year = {2024},
}
""",
    )
