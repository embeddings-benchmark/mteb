from mteb.abstasks.task_metadata import TaskMetadata

from ....abstasks.AbsTaskAnySTS import AbsTaskAnySTS


class ATEC(AbsTaskAnySTS):
    metadata = TaskMetadata(
        name="ATEC",
        dataset={
            "path": "C-MTEB/ATEC",
            "revision": "0f319b1142f28d00e055a6770f3f726ae9b7d865",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{raghu-etal-2021-end,
  abstract = {We propose a novel problem within end-to-end learning of task oriented dialogs (TOD), in which the dialog system mimics a troubleshooting agent who helps a user by diagnosing their problem (e.g., car not starting). Such dialogs are grounded in domain-specific flowcharts, which the agent is supposed to follow during the conversation. Our task exposes novel technical challenges for neural TOD, such as grounding an utterance to the flowchart without explicit annotation, referring to additional manual pages when user asks a clarification question, and ability to follow unseen flowcharts at test time. We release a dataset (FLODIAL) consisting of 2,738 dialogs grounded on 12 different troubleshooting flowcharts. We also design a neural model, FLONET, which uses a retrieval-augmented generation architecture to train the dialog agent. Our experiments find that FLONET can do zero-shot transfer to unseen flowcharts, and sets a strong baseline for future research.},
  address = {Online and Punta Cana, Dominican Republic},
  author = {Raghu, Dinesh  and
Agarwal, Shantanu  and
Joshi, Sachindra  and
{Mausam}},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2021.emnlp-main.357},
  editor = {Moens, Marie-Francine  and
Huang, Xuanjing  and
Specia, Lucia  and
Yih, Scott Wen-tau},
  month = nov,
  pages = {4348--4366},
  publisher = {Association for Computational Linguistics},
  title = {End-to-End Learning of Flowchart Grounded Task-Oriented Dialogs},
  url = {https://aclanthology.org/2021.emnlp-main.357},
  year = {2021},
}
""",
    )

    min_score = 0
    max_score = 1


class BQ(AbsTaskAnySTS):
    metadata = TaskMetadata(
        name="BQ",
        dataset={
            "path": "C-MTEB/BQ",
            "revision": "e3dda5e115e487b39ec7e618c0c6a29137052a55",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{xiao2024cpackpackagedresourcesadvance,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
  url = {https://arxiv.org/abs/2309.07597},
  year = {2024},
}
""",
    )

    min_score = 0
    max_score = 1


class LCQMC(AbsTaskAnySTS):
    metadata = TaskMetadata(
        name="LCQMC",
        dataset={
            "path": "C-MTEB/LCQMC",
            "revision": "17f9b096f80380fce5ed12a9be8be7784b337daf",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{xiao2024cpackpackagedresourcesadvance,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
  url = {https://arxiv.org/abs/2309.07597},
  year = {2024},
}
""",
    )

    min_score = 0
    max_score = 1


class PAWSX(AbsTaskAnySTS):
    metadata = TaskMetadata(
        name="PAWSX",
        dataset={
            "path": "mteb/PAWSX",
            "revision": "bd129d4230ee0551b5469c566bced8da75abae0a",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{xiao2024cpackpackagedresourcesadvance,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
  url = {https://arxiv.org/abs/2309.07597},
  year = {2024},
}
""",
    )

    min_score = 0
    max_score = 1


class STSB(AbsTaskAnySTS):
    metadata = TaskMetadata(
        name="STSB",
        dataset={
            "path": "C-MTEB/STSB",
            "revision": "0cde68302b3541bb8b3c340dc0644b0b745b3dc0",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=None,
        domains=[],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{xiao2024cpackpackagedresourcesadvance,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
  url = {https://arxiv.org/abs/2309.07597},
  year = {2024},
}
""",
    )

    min_score = 0
    max_score = 5


class AFQMC(AbsTaskAnySTS):
    metadata = TaskMetadata(
        name="AFQMC",
        dataset={
            "path": "C-MTEB/AFQMC",
            "revision": "b44c3b011063adb25877c13823db83bb193913c4",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{raghu-etal-2021-end,
  abstract = {We propose a novel problem within end-to-end learning of task oriented dialogs (TOD), in which the dialog system mimics a troubleshooting agent who helps a user by diagnosing their problem (e.g., car not starting). Such dialogs are grounded in domain-specific flowcharts, which the agent is supposed to follow during the conversation. Our task exposes novel technical challenges for neural TOD, such as grounding an utterance to the flowchart without explicit annotation, referring to additional manual pages when user asks a clarification question, and ability to follow unseen flowcharts at test time. We release a dataset (FLODIAL) consisting of 2,738 dialogs grounded on 12 different troubleshooting flowcharts. We also design a neural model, FLONET, which uses a retrieval-augmented generation architecture to train the dialog agent. Our experiments find that FLONET can do zero-shot transfer to unseen flowcharts, and sets a strong baseline for future research.},
  address = {Online and Punta Cana, Dominican Republic},
  author = {Raghu, Dinesh  and
Agarwal, Shantanu  and
Joshi, Sachindra  and
{Mausam}},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2021.emnlp-main.357},
  editor = {Moens, Marie-Francine  and
Huang, Xuanjing  and
Specia, Lucia  and
Yih, Scott Wen-tau},
  month = nov,
  pages = {4348--4366},
  publisher = {Association for Computational Linguistics},
  title = {End-to-End Learning of Flowchart Grounded Task-Oriented Dialogs},
  url = {https://aclanthology.org/2021.emnlp-main.357},
  year = {2021},
}
""",
    )

    min_score = 0
    max_score = 1


class QBQTC(AbsTaskAnySTS):
    metadata = TaskMetadata(
        name="QBQTC",
        dataset={
            "path": "C-MTEB/QBQTC",
            "revision": "790b0510dc52b1553e8c49f3d2afb48c0e5c48b7",
        },
        description="",
        reference="https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    min_score = 0
    max_score = 2
