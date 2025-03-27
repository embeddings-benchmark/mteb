from __future__ import annotations

from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from functools import partial
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

class MoAGate(nn.Module):
    def __init__(self, num_adaptors, hidden_dim):
        super().__init__()
        self.routing_vectors = nn.Parameter(
                torch.empty(num_adaptors, hidden_dim, dtype=torch.float32),
                requires_grad=False
            )
    def forward(self, hidden_states):
        if self.routing_vectors.device == torch.device('cpu'):
            self.routing_vectors = self.routing_vectors.to(hidden_states.device)
        hidden_states = hidden_states.unsqueeze(1)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        distances = torch.cdist(hidden_states, self.routing_vectors)

        _, cluster_indices = torch.min(distances, dim=1)
        cluster_indices = cluster_indices.view(-1, 1)

        topk_indices = cluster_indices
        topk_indices = torch.zeros_like(topk_indices, device=hidden_states.device)
        topk_weights = torch.ones_like(topk_indices, device=hidden_states.device)

        return topk_indices, topk_weights

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class MixtureOfAdaptors(nn.Module):
    def __init__(self, num_adaptors, hidden_dim):
        super().__init__()
        self.adaptors = nn.ModuleList([
            LinearLayer(input_dim=hidden_dim, output_dim=hidden_dim)
            for _ in range(num_adaptors)
        ])
        self.gate = MoAGate(num_adaptors, hidden_dim)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            hidden_states = inputs['sentence_embedding']
        else:
            hidden_states = inputs

        residual = hidden_states
        original_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_indices = topk_indices.view(-1)
        output = self.moa_inference(hidden_states, flat_topk_indices, topk_weights.view(-1, 1)).view(*original_shape)

        if isinstance(inputs, dict):
            inputs['sentence_embedding'] = output
            return inputs
        return output

    @torch.no_grad()
    def moa_inference(self, x, flat_adaptor_indices, flat_adaptor_weights):
        adaptor_cache = torch.zeros_like(x)
        sorted_indices = flat_adaptor_indices.argsort()
        tokens_per_adaptor = flat_adaptor_indices.bincount().cpu().numpy().cumsum(0)
        token_indices = sorted_indices
        for i, end_idx in enumerate(tokens_per_adaptor):
            start_idx = 0 if i == 0 else tokens_per_adaptor[i-1]
            if start_idx == end_idx:
                continue
            adaptor = self.adaptors[i]
            adaptor_token_indices = token_indices[start_idx:end_idx]
            adaptor_tokens = x[adaptor_token_indices]
            adaptor_output = adaptor(adaptor_tokens)
            adaptor_output.mul_(flat_adaptor_weights[sorted_indices[start_idx:end_idx]])
            adaptor_cache.scatter_reduce_(
                0,
                adaptor_token_indices.view(-1, 1).repeat(1, x.shape[-1]),
                adaptor_output,
                reduce='sum'
            )
        return adaptor_cache

class YuanWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('IEITYuan/Yuan-embedding-1.0', trust_remote_code=True)
        adaptor = MixtureOfAdaptors(5, 1792)

        model_path = snapshot_download(
            repo_id="OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
            local_dir="./Ops-MoA-Yuan-embedding-1.0"
        )
        adaptor.load_state_dict(torch.load(f"Ops-MoA-Yuan-embedding-1.0/yuan-adaptors.pth"))

        self.model.add_module('adaptor', adaptor)
        self.output_dim = 1536

    def encode(
         self,
         sentences: list[str],
         **kwargs
    ) -> np.ndarray:
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, :self.output_dim]

class ConanWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('TencentBAC/Conan-embedding-v1', trust_remote_code=True)
        adaptor = MixtureOfAdaptors(5, 1792)

        model_path = snapshot_download(
            repo_id="OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
            local_dir="./Ops-MoA-Conan-embedding-v1"
        )
        adaptor.load_state_dict(torch.load(f"Ops-MoA-Conan-embedding-v1/conan-adaptors.pth"))

        self.model.add_module('adaptor', adaptor)
        self.output_dim = 1536

    def encode(
         self,
         sentences: list[str],
         **kwargs
    ) -> np.ndarray:
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, :self.output_dim]

ops_moa_conan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    revision="cd42de6d61c103047b7bcd780ef0dbaa9a9d0472",
    release_date="2025-03-26",
    languages=["zho_Hans"],
    loader=partial(
         ConanWrapper,
    ),
    n_parameters=343 * 1e6,
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
    superseded_by=None,
)

ops_moa_yuan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    revision="09b8857bbd74f189d9bc45bf59adaf34f9378e17",
    release_date="2025-03-26",
    languages=["zho_Hans"],
    loader=partial(
         YuanWrapper,
    ),
    n_parameters=343 * 1e6,
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets=None,
    superseded_by=None,
)
