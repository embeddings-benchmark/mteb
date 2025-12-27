from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import BatchedInput, PromptType

from .rerankers_custom import RerankerWrapper

LISTCONRANKER_CITATION = """@article{liu2025listconranker,
  title={ListConRanker: A Contrastive Text Reranker with Listwise Encoding},
  author={Liu, Junlong and Ma, Yue and Zhao, Ruihui and Zheng, Junhao and Ma, Qianli and Kang, Yangyang},
  journal={arXiv preprint arXiv:2501.07111},
  year={2025}
}"""


class ListConRanker(RerankerWrapper):
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        super().__init__(model_name_or_path, **kwargs)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options

        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        queries = [text for batch in inputs1 for text in batch["query"]]
        passages = [text for batch in inputs2 for text in batch["text"]["text"]]

        assert len(queries) == len(passages)

        final_scores = []
        query = queries[0]
        tmp_passages = []
        if kwargs.get("traditional_inference"):
            for q, p in zip(queries, passages):
                if query == q:
                    tmp_passages.append(p)
                else:
                    query_passages_tuples = [[query] + tmp_passages]
                    scores = self.model.multi_passage(query_passages_tuples)
                    final_scores += scores
                    query = q
                    tmp_passages = [p]
            if len(tmp_passages) > 0:
                query_passages_tuples = [[query] + tmp_passages]
                scores = self.model.multi_passage(query_passages_tuples)
                final_scores += scores
        else:
            for q, p in zip(queries, passages):
                if query == q:
                    tmp_passages.append(p)
                else:
                    query_passages = [query] + tmp_passages
                    scores = self.model.multi_passage_in_iterative_inference(
                        query_passages
                    )
                    final_scores += scores
                    query = q
                    tmp_passages = [p]
            if len(tmp_passages) > 0:
                query_passages = [query] + tmp_passages
                scores = self.model.multi_passage_in_iterative_inference(query_passages)
                final_scores += scores

        assert len(final_scores) == len(queries), (
            f"Expected {len(queries)} scores, got {len(final_scores)}"
        )

        return final_scores


listconranker_training_datasets = {
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
    "MMarcoReranking",
    "T2Reranking",
    # 'Huatuo26M-Lite': ['train'],
    # 'MARC': ['train'],
    # 'XL-sum-chinese_simplified': ['train'],
    # 'CSL': ['train'],
}

listconranker = ModelMeta(
    loader=ListConRanker,
    loader_kwargs=dict(
        fp_options="float16",
    ),
    name="ByteDance/ListConRanker",
    model_type=["cross-encoder"],
    languages=["zho-Hans"],
    open_weights=True,
    revision="95ae6a5f422a916bc36520f0f3e198e7d91520a0",
    release_date="2024-12-11",
    n_parameters=401_000_000,
    memory_usage_mb=1242,
    similarity_fn_name="cosine",
    training_datasets=listconranker_training_datasets,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/ByteDance/ListConRanker",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    citation=LISTCONRANKER_CITATION,
)
