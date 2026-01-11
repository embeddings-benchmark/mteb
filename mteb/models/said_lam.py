"""
SAID-LAM model for MTEB.

Linear Attention Memory (LAM) with SAID Crystalline Attention (SCA) - BETA.
Achieves perfect recall via deterministic attention with 0.0% signal loss.

Organization: Said-Research
Reference: https://saidhome.ai

Free tier: 12K tokens
Licensed tier: 32K tokens
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class SAIDLAMEncoder(AbsEncoder):
    """
    SAID-LAM Encoder implementing MTEB's AbsEncoder protocol.
    
    Linear Attention Memory (LAM) with SAID Crystalline Attention (SCA).
    384-dimensional embeddings with O(n) linear complexity.
    
    Args:
        model_name: Model name (e.g., "Said-Research/SAID-LAM-v1")
        revision: Model revision (default: "main")
        device: Device to run on (default: auto-detect)
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SAID-LAM encoder."""
        self.model_name = model_name
        self.revision = revision or "main"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._embedding_dim = 384
        self._model = None
        
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize the LAM model."""
        # Add LAM package paths
        lam_paths = [
            Path("/workspace/LAM/lam_package"),  # Development path
            Path.home() / ".cache" / "said-lam",  # User cache
            Path(__file__).parent.parent,  # Relative to this file
        ]
        
        for path in lam_paths:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        
        # Model path resolution - look for trained model
        model_paths = [
            Path("/workspace/LAM/best"),  # Default development path
            Path.home() / ".cache" / "huggingface" / "hub" / self.model_name.replace("/", "--"),
        ]
        
        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = p
                break
        
        try:
            from lam import LAM
            self._model = LAM(str(model_path) if model_path else None, device=self.device)
            logger.info(f"SAID-LAM model loaded on {self.device}")
        except ImportError as e:
            raise ImportError(
                f"LAM package not found. Please install from https://saidhome.ai: {e}"
            )
    
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode inputs to embeddings.
        
        Args:
            inputs: DataLoader with batches of text
            task_metadata: Task metadata
            hf_split: HuggingFace split name
            hf_subset: HuggingFace subset name
            prompt_type: Prompt type (query or passage)
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of embeddings with shape (n_samples, 384)
        """
        all_texts = []
        for batch in inputs:
            texts = self._extract_texts(batch)
            all_texts.extend(texts)
        
        if not all_texts:
            return np.array([]).reshape(0, self._embedding_dim)
        
        # Apply clustering prefix for clustering tasks
        task_name = str(getattr(task_metadata, 'name', '')).lower()
        if 'cluster' in task_name:
            all_texts = ["cluster: " + t for t in all_texts]
        
        with torch.no_grad():
            embeddings = self._model.encode(
                all_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        
        return embeddings
    
    def _extract_texts(self, batch: Any) -> list[str]:
        """Extract texts from batch dictionary."""
        if isinstance(batch, dict):
            # Handle MTEB batch format
            for key in ['text', 'body', 'query', 'sentence']:
                if key in batch:
                    vals = batch[key] if isinstance(batch[key], list) else [batch[key]]
                    # Handle title + body combination
                    titles = batch.get('title', [''] * len(vals))
                    if isinstance(titles, str):
                        titles = [titles] * len(vals)
                    if key in ('body', 'text') and any(titles):
                        return [f"{t} {v}".strip() for t, v in zip(titles, vals)]
                    return [str(v) for v in vals]
        elif isinstance(batch, (list, tuple)):
            if batch and isinstance(batch[0], dict):
                return sum([self._extract_texts(d) for d in batch], [])
            return [str(s) for s in batch]
        return [str(batch)]
    
    def similarity(
        self, 
        emb1: np.ndarray | torch.Tensor, 
        emb2: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity matrix between embeddings."""
        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2)
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        return torch.mm(emb1, emb2.T)
    
    def similarity_pairwise(
        self, 
        emb1: np.ndarray | torch.Tensor, 
        emb2: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cosine similarity between embeddings."""
        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2)
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        return (emb1 * emb2).sum(dim=-1)


# =============================================================================
# Model Registration
# =============================================================================

said_lam_v1 = ModelMeta(
    name="Said-Research/SAID-LAM-v1",
    revision="main",
    release_date="2026-01-01",
    languages=["eng"],
    loader=SAIDLAMEncoder,  # Class reference (not instance or function)
    loader_kwargs=dict(
        # Additional kwargs passed to SAIDLAMEncoder.__init__
    ),
    n_parameters=22_000_000,
    memory_usage_mb=100,
    max_tokens=32000,  # Licensed tier (Free: 12K)
    embed_dim=384,
    license="not specified",  # Proprietary
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    framework=["PyTorch"],
    reference="https://saidhome.ai",
    similarity_fn_name="cosine",
    use_instructions=False,
)
