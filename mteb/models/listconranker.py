from mteb.model_meta import ModelMeta
from mteb.models.rerankers_custom import RerankerWrapper
from mteb.abstasks import TaskMetadata
from mteb.types import BatchedInput, PromptType
from torch.utils.data import DataLoader
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from typing import Any


class ListConRanker(RerankerWrapper):
    def __init__(
            self,
            model_name_or_path: str = None,
            **kwargs
    ) -> None:
        super().__init__(model_name_or_path, **kwargs)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
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
        
        instructions = None
        queries = [text for batch in inputs1 for text in batch["query"]]
        passages = [text for batch in inputs2 for text in batch["text"]['text']]
        
        assert len(queries) == len(passages)

        final_scores = []
        query = queries[0]
        tmp_passages = []
        if 'traditional_inference' in kwargs and kwargs['traditional_inference']:
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
                    scores = self.model.multi_passage_in_iterative_inference(query_passages)
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
        

listconranker = ModelMeta(
    loader=ListConRanker,
    loader_kwargs=dict(
        fp_options="float16",
    ),
    name='ByteDance/ListConRanker',
    languages=['zho-Hans'],
    open_weights=True,
    revision='95ae6a5f422a916bc36520f0f3e198e7d91520a0',
    release_date='2025-6-20',
    n_parameters=401_000_000,
    memory_usage_mb=None,
    similarity_fn_name=None,
    training_datasets=None,
    embed_dim=1024,
    license='mit',
    max_tokens=512,
    reference='https://huggingface.co/ByteDance/ListConRanker',
    framework=['PyTorch'],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    is_cross_encoder=True
)