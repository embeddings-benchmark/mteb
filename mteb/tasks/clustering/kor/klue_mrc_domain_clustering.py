from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class KlueMrcDomainClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="KlueMrcDomainClustering",
        description="this dataset is a processed and redistributed version of the KLUE-MRC dataset. Domain: Game / Media / Automotive / Finance / Real Estate / Education",
        reference="https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_context_domain",
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="v_measure",
        dataset={
            "path": "mteb/KlueMrcDomainClustering",
            "revision": "058684c3d416fff6681be45864aed41b75e334b2",
        },
        date=("2016-01-01", "2020-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{park2021klue,
  archiveprefix = {arXiv},
  author = {Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
  eprint = {2105.09680},
  primaryclass = {cs.CL},
  title = {KLUE: Korean Language Understanding Evaluation},
  year = {2021},
}
""",
        prompt="Identify the topic or theme of the given texts",
    )
