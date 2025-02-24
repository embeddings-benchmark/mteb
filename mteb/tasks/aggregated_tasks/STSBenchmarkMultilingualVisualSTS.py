from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.overview import get_tasks

task_list: list[AbsTask] = [
    get_tasks(tasks=["STSBenchmarkMultilingualVisualSTS"], languages=["eng"])
]


class STSBenchmarkMultilingualVisualSTSEng(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="VisualSTS-b-Eng",
        description="STSBenchmarkMultilingualVisualSTS English only.",
        reference="https://arxiv.org/abs/2402.08183/",
        tasks=task_list,
        main_score="cosine_spearman",
        type="VisualSTS(eng)",
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )


task_list_multi: list[AbsTask] = [
    get_tasks(
        tasks=["STSBenchmarkMultilingualVisualSTS"],
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
    )
]


class STSBenchmarkMultilingualVisualSTSMultilingual(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="VisualSTS-b-Multilingual",
        description="STSBenchmarkMultilingualVisualSTS multilingual.",
        reference="https://arxiv.org/abs/2402.08183/",
        tasks=task_list_multi,
        main_score="cosine_spearman",
        type="VisualSTS(multi)",
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )
