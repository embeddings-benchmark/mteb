"""Reproducible R70 task-routed embedding encoder (no test labels used)."""
from __future__ import annotations

from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

PROMPTS = {
    "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    "document": " ",
    "AILACasedocs-query": "Instruct: Given a legal case scenario, retrieve the most relevant case documents\nQuery:",
    "AILAStatutes-query": "Instruct: Given a legal scenario, retrieve the most relevant statute documents\nQuery:",
    "GerDaLIRSmall-query": "Instruct: Given a passage from a German court decision, retrieve the legal case documents cited by that passage.\nQuery: ",
    "LeCaRDv2-query": "Instruct: Retrieve text based on user query.\nQuery:",
    "LegalBenchConsumerContractsQA-query": "Instruct: Retrieve text based on user query.\nQuery:",
    "LegalBenchCorporateLobbying-query": "Instruct: Retrieve text based on user query.\nQuery:",
    "LegalQuAD-query": "Instruct: Given a legal question, retrieve relevant legal documents that answer the question\nQuery:",
    "LegalSummarization-query": "Instruct: Given a query, retrieve relevant legal documents for summarization\nQuery:",
}

class NGALawEncoder(SentenceTransformerEncoderWrapper):
    def __init__(self, model_name, revision=None, device=None, **kwargs):
        import numpy as np
        from huggingface_hub import hf_hub_download

        super().__init__(
            model_name, revision=revision, device=device, model_prompts=PROMPTS,
            model_kwargs={"torch_dtype": "bfloat16"}, **kwargs,
        )
        self.model.max_seq_length = 32768
        self.adapters = {}
        for name in ("gerda_query", "gerda_symmetric", "lecard_symmetric"):
            path = hf_hub_download(model_name, f"adapters/{name}.npz", revision=revision)
            with np.load(path, allow_pickle=False) as weights:
                self.adapters[name] = (weights["down"].astype(np.float32), weights["up"].astype(np.float32))

    def _adapt(self, values, name):
        import numpy as np

        down, up = self.adapters[name]
        values = np.asarray(values, dtype=np.float32)
        values = values + (values @ down) @ up
        return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-12)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        result = super().encode(
            inputs, task_metadata=task_metadata, hf_split=hf_split, hf_subset=hf_subset,
            prompt_type=prompt_type, **kwargs,
        )
        if task_metadata.name == "LeCaRDv2":
            return self._adapt(result, "lecard_symmetric")
        if task_metadata.name == "GerDaLIRSmall":
            if "query" in str(prompt_type).lower():
                result = self._adapt(result, "gerda_query")
            return self._adapt(result, "gerda_symmetric")
        return result


from mteb.models.model_meta import ModelMeta
from mteb.types import OutputDType

nga_law_r70 = ModelMeta(
    name="NGA-KR/law-embed-0.6b-r70",
    revision="e36d2b0b22d9502a55ef9059820975dbd1c9d65d",
    loader=NGALawEncoder,
    release_date="2026-10-27",
    languages=["eng-Latn", "deu-Latn", "zho-Hans"],
    n_parameters=595_874_816,
    n_embedding_parameters=155_309_056,
    memory_usage_mb=1137,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    open_weights=True,
    use_instructions=True,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/NGA-KR/law-embed-0.6b-r70",
    similarity_fn_name="cosine",
    training_datasets={"GerDaLIRSmall", "LeCaRDv2"},
    adapted_from="litillabs/litil-embed-0.6b",
    modalities=["text"],
    model_type=["dense"],
    output_dtypes=OutputDType.BF16,
    public_training_code="https://huggingface.co/NGA-KR/law-embed-0.6b-r70/tree/main",
    public_training_data=None,
)
