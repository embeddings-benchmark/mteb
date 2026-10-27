from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_RT1_BIBTEX = r"""
@inproceedings{brohan2022rt1,
  author = {Brohan, Anthony and Brown, Noah and Carbajal, Justice and Chebotar, Yevgen and Dabis, Joseph and Finn, Chelsea and Gopalakrishnan, Keerthana and Hausman, Karol and Herzog, Alexander and Hsu, Jasmine and Ibarz, Julian and Ichter, Brian and Irpan, Alex and Jackson, Tomas and Jesmonth, Sally and Joshi, Nikhil and Julian, Ryan and Kalashnikov, Dmitry and Kuang, Yuheng and Leal, Isabel and Lee, Kuang-Huei and Levine, Sergey and Lu, Yao and Malla, Utsav and Manjunath, Deeksha and Mordatch, Igor and Nachum, Ofir and Parada, Carolina and Peralta, Jodilyn and Perez, Emily and Pertsch, Karl and Quiambao, Jornell and Rao, Kanishka and Ryoo, Michael and Salazar, Grecia and Sanketi, Pannag and Sayed, Kevin and Singh, Jaspiar and Sontakke, Sumedh and Stone, Austin and Tan, Clayton and Tran, Huong and Vanhoucke, Vincent and Vega, Steve and Vuong, Quan and Xia, Fei and Xiao, Ted and Xu, Peng and Xu, Sichun and Yu, Tianhe and Zitkovich, Brianna},
  booktitle = {Robotics: Science and Systems XIX},
  collection = {RSS2023},
  doi = {10.15607/rss.2023.xix.025},
  month = {July},
  publisher = {Robotics: Science and Systems Foundation},
  series = {RSS2023},
  title = {RT-1: Robotics Transformer for Real-World Control at Scale},
  url = {https://doi.org/10.15607/rss.2023.xix.025},
  year = {2023},
}
"""

_RT1_DESCRIPTION_TAIL = (
    "Built from the RT-1 robot demonstration data (87,212 real-world "
    "Google Robot manipulation episodes, 578 unique language "
    "instructions): episodes are filtered to 5-100 s, 150 instructions "
    "are kept by even subsampling over the alphabetically sorted "
    "instruction list (spreading skills such as close/move/open/pick/"
    "place), and 10 episodes are sampled evenly per kept instruction."
)


class RT1T2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RT1T2VRetrieval",
        description=(
            "Text-to-video retrieval over real-world robot manipulation: "
            "given a language instruction, retrieve manipulation videos of "
            "episodes performing that instruction. Relevance is "
            "instruction-level and multi-positive: a query matches every "
            "corpus video of the same instruction (150 queries, 1,500 "
            "videos). " + _RT1_DESCRIPTION_TAIL
        ),
        reference="https://robotics-transformer1.github.io/",
        dataset={
            "path": "ZhixuLi/RT1-T2V",
            "revision": "8cfc2e1eb9e73c8f0f9799fe9832e5288d864ac2",
        },
        type="Any2AnyRetrieval",
        category="t2v",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_RT1_BIBTEX,
        prompt={
            "query": (
                "Retrieve robot manipulation videos of episodes performing "
                "the given instruction."
            )
        },
    )


class RT1V2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RT1V2TRetrieval",
        description=(
            "Video-to-text retrieval over real-world robot manipulation: "
            "given a manipulation episode video, retrieve the language "
            "instruction it performs among all 578 unique instructions of "
            "the source dataset (most corpus texts are distractors; each "
            "query has exactly one relevant text; 300 queries, 2 per kept "
            "instruction). " + _RT1_DESCRIPTION_TAIL
        ),
        reference="https://robotics-transformer1.github.io/",
        dataset={
            "path": "ZhixuLi/RT1-V2T",
            "revision": "af3155fd08e9c208e37102ca1f4ffb0c11d53784",
        },
        type="Any2AnyRetrieval",
        category="v2t",
        modalities=["text", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Robotics", "Scene"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_RT1_BIBTEX,
        prompt={
            "query": (
                "Retrieve the language instruction that the robot "
                "manipulation video performs."
            )
        },
    )
