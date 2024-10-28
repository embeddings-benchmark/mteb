from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class Flickr30kT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="Flickr30kT2IRetrieval",
        description="Retrieve images based on captions.",
        reference="https://www.semanticscholar.org/paper/From-image-descriptions-to-visual-denotations%3A-New-Young-Lai/44040913380206991b1991daf1192942e038fe31",
        dataset={
            "path": "JamieSJS/flickr30k",
            "revision": "a4cf34ac79215f9e2cd6a10342d84f606fc41cc3",
        },
        type="Any2AnyRetrieval",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@article{Young2014FromID,
  title={From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
  author={Peter Young and Alice Lai and Micah Hodosh and J. Hockenmaier},
  journal={Transactions of the Association for Computational Linguistics},
  year={2014},
  volume={2},
  pages={67-78},
  url={https://api.semanticscholar.org/CorpusID:3104920}
}""",
        descriptive_stats={
            "n_samples": {"default": 31014},  # qrels
        },
    )

    def load_data(self, **kwargs):
        super().load_data(**kwargs)
        # swap corpus and query
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            self.queries[split], self.corpus[split] = (
                self.corpus[split],
                self.queries[split],
            )
            self.relevant_docs[split] = {
                cid: {qid: score}
                for qid, cid_score in self.relevant_docs[split].items()
                for cid, score in cid_score.items()
            }
