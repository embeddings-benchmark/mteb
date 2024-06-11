from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class MindSmallReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="MindSmallReranking",
        description="Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
        reference="https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
        hf_hub_name="mteb/mind_small",
        dataset={
            "path": "mteb/mind_small",
            "revision": "59042f120c80e8afa9cdbb224f67076cec0fc9a7",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{wu-etal-2020-mind, title = "{MIND}: A Large-scale Dataset for News 
        Recommendation", author = "Wu, Fangzhao  and Qiao, Ying  and Chen, Jiun-Hung  and Wu, Chuhan  and Qi, 
        Tao  and Lian, Jianxun  and Liu, Danyang  and Xie, Xing  and Gao, Jianfeng  and Wu, Winnie  and Zhou, Ming", 
        editor = "Jurafsky, Dan  and Chai, Joyce  and Schluter, Natalie  and Tetreault, Joel", booktitle = 
        "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics", month = jul, 
        year = "2020", address = "Online", publisher = "Association for Computational Linguistics", 
        url = "https://aclanthology.org/2020.acl-main.331", doi = "10.18653/v1/2020.acl-main.331", 
        pages = "3597--3606", abstract = "News recommendation is an important technique for personalized news 
        service. Compared with product and movie recommendations which have been comprehensively studied, 
        the research on news recommendation is much more limited, mainly due to the lack of a high-quality benchmark 
        dataset. In this paper, we present a large-scale dataset named MIND for news recommendation. Constructed from 
        the user click logs of Microsoft News, MIND contains 1 million users and more than 160k English news 
        articles, each of which has rich textual content such as title, abstract and body. We demonstrate MIND a good 
        testbed for news recommendation through a comparative study of several state-of-the-art news recommendation 
        methods which are originally developed on different proprietary datasets. Our results show the performance of 
        news recommendation highly relies on the quality of news content understanding and user interest modeling. 
        Many natural language processing techniques such as effective text representation methods and pre-trained 
        language models can effectively improve the performance of news recommendation. The MIND dataset will be 
        available at https://msnews.github.io}.", }""",
        n_samples={"test": 107968},
        avg_character_length={"test": 70.9},
    )
