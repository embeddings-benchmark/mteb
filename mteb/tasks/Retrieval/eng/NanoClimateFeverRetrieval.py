from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from mteb import MTEB


class NanoClimateFeverRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoClimateFeverRetrieval",
        description="NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.",
        reference="https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6",
        dataset={
            "path": "zeta-alpha-ai/NanoClimateFEVER",
            "revision": "7bda449ec7e1965490bb862bae3d8c0f419b5611de561c7fd4ce7d7274b843ac",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@misc{diggelmann2021climatefever,
      title={CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims}, 
      author={Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
      year={2021},
      eprint={2012.00614},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus = load_dataset("zeta-alpha-ai/NanoClimateFEVER", 'corpus')
        self.queries = load_dataset("zeta-alpha-ai/NanoClimateFEVER", 'queries')
        self.relevant_docs = load_dataset("zeta-alpha-ai/NanoClimateFEVER", 'qrels')

        self.corpus = {
            split: {
                sample['_id']: {'_id': sample['_id'], 'text': sample['text']}
                for sample in self.corpus[split]
            }
            for split in self.corpus
        }

        self.queries = {
            split: {
                sample['_id']: sample['text']
                for sample in self.queries[split]
            }
            for split in self.queries
        }

        self.relevant_docs = {
            split: {
                sample['query-id']: {
                    sample['corpus-id']: 1  # Assuming a score of 1 for relevant documents
                }
                for sample in self.relevant_docs[split]
            }
            for split in self.relevant_docs
        }

        # print("corpus")
        # print(self.corpus)
        # print("queries")
        # print(self.queries)
        # print("relevant_docs")
        # print(self.relevant_docs)
        self.data_loaded = True