from __future__ import annotations

from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.Image.VisualSTS import STS17MultilingualVisualSTS

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
        annotations_creators="human-annotated",
        dialect=[""],
        eval_langs={
            "en-en": ["eng-Latn"]
        },  # rely on subsets to filter scores in TaskResults.get_score_fast().
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="VisualSTS(eng)",
        eval_splits=["test"],
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )


task_list_sts17_multi: list[AbsTask] = [
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
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[""],
        sample_creation="rendered",
        main_score="cosine_spearman",
        type="VisualSTS(multi)",
        eval_splits=["test"],
        eval_langs={  # rely on subsets to filter scores in TaskResults.get_score_fast().
            "ko-ko": ["kor-Hang"],
            "ar-ar": ["ara-Arab"],
            "en-ar": ["eng-Latn", "ara-Arab"],
            "en-de": ["eng-Latn", "deu-Latn"],
            "en-tr": ["eng-Latn", "tur-Latn"],
            "es-en": ["spa-Latn", "eng-Latn"],
            "es-es": ["spa-Latn"],
            "fr-en": ["fra-Latn", "eng-Latn"],
            "it-en": ["ita-Latn", "eng-Latn"],
            "nl-en": ["nld-Latn", "eng-Latn"],
        },
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )
