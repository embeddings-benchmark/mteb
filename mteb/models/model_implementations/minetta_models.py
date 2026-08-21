from __future__ import annotations

import numpy as np

from mteb.models import (
    ModelMeta,
    SentenceTransformerEncoderWrapper,
    sentence_transformers_loader,
)
from mteb.models.model_meta import ScoringFunction

nemotron_3_embed_8b_legal = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs={"model_kwargs": {"torch_dtype": "bfloat16"}},
    name="minetta/nemotron-3-embed-8b-legal",
    revision="70d7e152f3a5e676478c9f947b1e23c4ba755019",
    release_date="2026-07-25",
    languages=["eng-Latn"],
    n_parameters=7_952_700_148,
    n_embedding_parameters=536_870_912,
    memory_usage_mb=15168,
    max_tokens=32768,
    embed_dim=4096,
    license="https://huggingface.co/minetta/nemotron-3-embed-8b-legal/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/minetta/nemotron-3-embed-8b-legal",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from="nvidia/Nemotron-3-Embed-8B-BF16",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
)

_HEADS = {
    "MedrxivClusteringS2S.v2": "clustering_head/head_s2s.npz",
    "MedrxivClusteringP2P.v2": "clustering_head/head_p2p.npz",
}


def _medical_loader(model_name: str, revision: str, **kwargs):
    """Apply the clustering projection stored in the model repo, pass other tasks through.

    Both projections are loaded once here, when the model is built, so nothing is fetched
    during evaluation. Keying on task name is what model_prompts does
    (abs_encoder.py:208), and jasper_models.py and harrier_models.py key on these same two
    names. The map covers only the tasks the projection was measured on. Output stays
    4096-dimensional, so embed_dim holds for every task.
    """
    from huggingface_hub import hf_hub_download

    encoder = SentenceTransformerEncoderWrapper(
        model=model_name, revision=revision, **kwargs
    )
    heads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task, path in _HEADS.items():
        z = np.load(hf_hub_download(model_name, path, revision=revision))
        heads[task] = (z["mu"].astype(np.float32), z["P"].astype(np.float32))
    inner_encode = encoder.encode

    def encode(inputs, *args, **kw):
        x = np.asarray(inner_encode(inputs, *args, **kw), dtype=np.float32)
        head = heads.get(getattr(kw.get("task_metadata"), "name", None))
        if head is None:
            return x
        mu, proj = head
        z = ((x - mu) @ proj.T) @ proj
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

    encoder.encode = encode
    return encoder


nemotron_3_embed_8b_medical = ModelMeta(
    loader=_medical_loader,
    # model_max_length matches the base entry in nvidia_models.py. The checkpoint advertises a
    # 32k window, but TRECCOVID and TRECCOVID-PL contain 122k-character documents, and running
    # those uncapped needs more memory than an 80GB card has.
    loader_kwargs={
        "model_kwargs": {"torch_dtype": "bfloat16"},
        "processor_kwargs": {"model_max_length": 4096},
    },
    name="minetta/nemotron-3-embed-8b-medical",
    revision="388e7e642e6d486762f4e49d97d756af8fb63723",
    release_date="2026-07-29",
    languages=["eng-Latn", "cmn-Hans", "pol-Latn"],
    n_parameters=7_952_683_008,
    n_embedding_parameters=536_870_912,
    memory_usage_mb=15168,
    max_tokens=32768,
    embed_dim=4096,
    license="https://huggingface.co/minetta/nemotron-3-embed-8b-medical/blob/main/LICENSE",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/minetta/nemotron-3-embed-8b-medical",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    # Empty set rather than None: nothing was trained on. The weights come from adapted_from
    # unchanged, and the projection is fitted on PubMed abstracts.
    training_datasets=set(),
    adapted_from="nvidia/Nemotron-3-Embed-8B-BF16",
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
)
