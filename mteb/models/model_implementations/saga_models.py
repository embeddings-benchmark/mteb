import numpy as np
import mteb
from typing import Any, Optional
from torch.utils.data import DataLoader
from mteb.cache import ResultCache
from mteb.models import ModelMeta, EncoderProtocol
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType

# 1. Your Custom Prompt Router
def my_custom_task_to_instruction(task_name: str, task_type: str) -> str:
    if task_type in ["STS", "Summarization", "Reranking", "BitextMining"]:
        if task_type == "BornholmBitextMining":
            return ""
        return "semantic similarity"
    if task_type == "Classification":
        if "Language" in task_name or "Lang" in task_name:
            return "clustering"
        return "classification"
    if task_type == "Retrieval":
        if any(name in task_name.lower() for name in ["hjerne", "faq", "quad", "question"]):
            return "question answering"
        return "retrieval"
    if task_type == "Clustering":
        return "clustering"
    return ""


class MyCustomModel(EncoderProtocol):
    def __init__(self, model_name: str, revision: str = None, device: str = None, **kwargs):
        from sentence_transformers import SentenceTransformer
        actual_path = kwargs.pop("model_path", model_name)
        self.model = SentenceTransformer(actual_path, device=device, **kwargs)
        self._meta: Optional[ModelMeta] = None
        self.device = device

    @property
    def mteb_model_meta(self) -> ModelMeta:
        if self._meta is None:
            raise ValueError("Model metadata has not been set yet.")
        return self._meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, value: ModelMeta):
        self._meta = value

    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: Optional[PromptType] = None,
        **kwargs: Any,
    ) -> np.ndarray:

        instruction = my_custom_task_to_instruction(task_metadata.name, task_metadata.type)
        all_embeddings = []

        for batch in inputs:
            texts = batch["text"]
            sentences_with_prompts = []

            if instruction == "retrieval":
                is_passage = (prompt_type == "passage")
                if is_passage:
                    sentences_with_prompts = [f"title: none | text: {s}" for s in texts]
                else:
                    sentences_with_prompts = [f"task: search result | query: {s}" for s in texts]
            elif instruction:
                 sentences_with_prompts = [f"task: {instruction} | query: {s}" for s in texts]
            else:
                 sentences_with_prompts = texts

            emb = self.model.encode(
                sentences_with_prompts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                **kwargs
            )
            all_embeddings.append(emb)

        if len(all_embeddings) > 0:
            return np.concatenate(all_embeddings, axis=0)
        return np.array([])

    def similarity(self, embeddings1, embeddings2):
        return self.model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1, embeddings2):
        return self.model.similarity_pairwise(embeddings1, embeddings2)


my_custom_model_meta = ModelMeta(
    name="nicher92/saga-embed_v1", 
    reference="https://huggingface.co/nicher92/saga-embed_v1",
    loader=MyCustomModel,
    loader_kwargs={}, 
    revision="3be07ac3d7c3e00e4402ae9285b23fcf8fda6735",
   release_date="2025-01-09",
    languages=["swe-Latn"],
    n_parameters=404_219_904
    memory_usage_mb=2167,
    license="mit",
    max_tokens=1024,
    embed_dim=1024,
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch"],
    similarity_fn_name="cosine",
    use_instructions=True,
    public_training_code=None,
    public_training_data=None, 
    training_datasets={
        "sentence-transformers/reddit",
        "sentence-transformers/xsum",
        "sentence-transformers/simple-wiki",
        "sentence-transformers/s2orc",
        "sentence-transformers/amazon-reviews",
        "sentence-transformers/gooaq",
        "sentence-transformers/paq",
        "sentence-transformers/stackexchange-duplicates",
        "sentence-transformers/wikipedia-sections",
        "stanfordnlp/snli",
        "tomaarsen/natural-questions-hard-negatives",
        "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
    },
    training_datasets=None,
)
