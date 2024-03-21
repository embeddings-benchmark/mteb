from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SciDocsReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SciDocsRR",
        description="Ranking of related scientific papers based on their title.",
        reference="https://allenai.org/data/scidocs",
        hf_hub_name="mteb/scidocs-reranking",
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="map",
        revision="d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        date="2020",
        form="written",
        domains=["Academic", "Non-fiction"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators=None,
        dialect=None,
        text_creation="found",
        bibtex_citation="""
@inproceedings{cohan-etal-2020-specter,
    title = "{SPECTER}: Document-level Representation Learning using Citation-informed Transformers",
    author = "Cohan, Arman  and
      Feldman, Sergey  and
      Beltagy, Iz  and
      Downey, Doug  and
      Weld, Daniel",
    editor = "Jurafsky, Dan  and
      Chai, Joyce  and
      Schluter, Natalie  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.207",
    doi = "10.18653/v1/2020.acl-main.207",
    pages = "2270--2282",
    abstract = "Representation learning is a critical ingredient for natural language processing systems. Recent Transformer language models like BERT learn powerful textual representations, but these models are targeted towards token- and sentence-level training objectives and do not leverage information on inter-document relatedness, which limits their document-level representation power. For applications on scientific documents, such as classification and recommendation, accurate embeddings of documents are a necessity. We propose SPECTER, a new method to generate document-level embedding of scientific papers based on pretraining a Transformer language model on a powerful signal of document-level relatedness: the citation graph. Unlike existing pretrained language models, Specter can be easily applied to downstream applications without task-specific fine-tuning. Additionally, to encourage further research on document-level models, we introduce SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation. We show that Specter outperforms a variety of competitive baselines on the benchmark.",
}
"""
    )
