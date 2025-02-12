from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.overview import get_tasks

task_list_sts17: list[AbsTask] = [
    get_tasks(tasks=["STSBenchmarkMultilingualVisualSTS"], languages=["eng"])
]


class STSBenchmarkMultilingualVisualSTSEng(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="STSBenchmarkMultilingualVisualSTSEng",
        description="STSBenchmarkMultilingualVisualSTS English only.",
        reference="https://arxiv.org/abs/2402.08183/",
        tasks=task_list_sts17,
        main_score="cosine_spearman",
        type="VisualSTS",
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )
