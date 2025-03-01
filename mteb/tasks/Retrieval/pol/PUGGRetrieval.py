from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PUGGRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PUGGRetrieval",
        description="Information Retrieval PUGG dataset for the Polish language.",
        reference="https://aclanthology.org/2024.findings-acl.652/",
        dataset={
            "path": "clarin-pl/PUGG_IR",
            "revision": "48eff464950391ce7a3d58f37355fceccf613725",
        },
        type="Retrieval",
        category="t2t",
        date=("2023-01-01", "2024-01-01"),
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="multiple",
        bibtex_citation=""""@inproceedings{sawczyn-etal-2024-developing,
    title = "Developing {PUGG} for {P}olish: A Modern Approach to {KBQA}, {MRC}, and {IR} Dataset Construction",
    author = "Sawczyn, Albert  and
      Viarenich, Katsiaryna  and
      Wojtasik, Konrad  and
      Domoga{\l}a, Aleksandra  and
      Oleksy, Marcin  and
      Piasecki, Maciej  and
      Kajdanowicz, Tomasz",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.652/",
    doi = "10.18653/v1/2024.findings-acl.652",
    pages = "10978--10996",
    abstract = "Advancements in AI and natural language processing have revolutionized machine-human language interactions, with question answering (QA) systems playing a pivotal role. The knowledge base question answering (KBQA) task, utilizing structured knowledge graphs (KG), allows for handling extensive knowledge-intensive questions. However, a significant gap exists in KBQA datasets, especially for low-resource languages. Many existing construction pipelines for these datasets are outdated and inefficient in human labor, and modern assisting tools like Large Language Models (LLM) are not utilized to reduce the workload. To address this, we have designed and implemented a modern, semi-automated approach for creating datasets, encompassing tasks such as KBQA, Machine Reading Comprehension (MRC), and Information Retrieval (IR), tailored explicitly for low-resource environments. We executed this pipeline and introduced the PUGG dataset, the first Polish KBQA dataset, and novel datasets for MRC and IR. Additionally, we provide a comprehensive implementation, insightful findings, detailed statistics, and evaluation of baseline models."
}

}""",
    )
