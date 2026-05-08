from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from sentence_transformers import SentenceTransformer

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

from .bidirlm_models import (
    BIDIRLM_CITATION,
    BIDIRLM_LANGUAGES,
    bidirlm_task_prompts,
    bidirlm_training_data,
)

BIDIRLM_OMNI_TRAINING_DATASETS = bidirlm_training_data | {
    "LAION-Audio-300M",
    "MS_COCO",
    "colpali_train_set",
    "natcap",
    "librispeech_asr",
}


TASK_PROMPTS: dict[str, str | dict[str, str]] = {
    **bidirlm_task_prompts,
    # MTEB tasks
    "ArguAna": {
        "query": "Given a claim, retrieve documents that support or refute the claim",
        "passage": "Given a claim, retrieve documents that support or refute the claim",
    },
    "CQADupstackGamingRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "CQADupstackUnixRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "passage": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    # MIEB tasks
    "AROCocoOrder": "Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.",
    "AROFlickrOrder": "Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.",
    "BLINKIT2IMultiChoice": "Retrieve images based on images and specific retrieval instructions.",
    "Country211ZeroShot": "Classifying images of 211 countries.",
    "CVBenchRelation": "decide the relation of the objects in the image.",
    "FER2013ZeroShot": "Classifying facial emotions.",
    "VidoreDocVQARetrieval": "Retrieve associated pages according to questions.",
    "VidoreShiftProjectRetrieval": "Retrieve associated pages according to questions.",
    "VidoreSyntheticDocQAAIRetrieval": "Retrieve associated pages according to questions.",
    "VidoreTabfquadRetrieval": "Retrieve associated pages according to questions.",
    "VidoreTatdqaRetrieval": "Retrieve associated pages according to questions.",
    "VQA2IT2TRetrieval": "Retrieve the correct answer for a question about an image.",
    "WebQAT2ITRetrieval": "Retrieve sources of information based on questions.",
    "WITT2IRetrieval": "Retrieve images based on multilingual descriptions.",
    "XM3600T2IRetrieval": "Retrieve images based on multilingual descriptions.",
    # MAEB tasks
    "CommonLanguageAgeDetection": "Age Classification. This is a stratified subsampled version of the original CommonLanguage dataset.",
    "CommonVoiceMini21T2ARetrieval": "Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
    "FSD2019Kaggle": "Multilabel Audio Classification.",
    "JamAltArtistA2ARetrieval": "Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset.",
    "JamAltLyricA2TRetrieval": "From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset.",
    "SpeechCommandsZeroshotv0.02": "Sound Classification/Keyword Spotting Dataset. This is a set of one-second audio clips containing a single spoken English word or background noise. These words are from a small set of commands such as 'yes', 'no', and 'stop' spoken by various speakers. With a total of 10 labels/commands for keyword spotting and a total of 30 labels for other auxiliary tasks.",
    "VehicleSoundClustering": "Clustering vehicle sounds recorded from smartphones (0 (car class), 1 (truck, bus and van class), 2 (motorcycle class)).",
    "VoxPopuliAccentPairClassification": "Classifying same or different regional accent of English.",
}

POOLING_DIM = 2048


class BidirLMOmniEncoder(AbsEncoder):
    """MTEB-compatible multimodal encoder for BidirLM-Omni (text / image / audio)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation: str = "eager",
        trust_remote_code: bool = True,
        max_text_length: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"attn_implementation": attn_implementation},
        )
        self.model.eval()
        self.max_text_length = max_text_length

        self.task_prompts = TASK_PROMPTS

    def _get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        task_type = task_metadata.type

        if task_type == "Summarization":
            return None

        entry = self.task_prompts.get(task_metadata.name)

        # Asymmetric retrieval: documents get no instruction unless the prompt
        # dict explicitly provides a "passage" key.
        if (
            task_metadata.simplified_task_type == "retrieval"
            and prompt_type == PromptType.document
            and not (isinstance(entry, dict) and "passage" in entry)
        ):
            return None

        if entry is not None:
            if isinstance(entry, dict):
                key = prompt_type.value if prompt_type else "query"
                instruction = entry.get(key, entry.get("query")) or None
            else:
                instruction = entry
            return instruction

        if task_type in {"STS", "PairClassification"}:
            return "Retrieve semantically similar text"
        if task_type == "BitextMining":
            return "Retrieve parallel sentences"

        return None

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Implements AbsEncoder.encode with multimodal support (text, image, audio).

        Builds conversation messages from whichever modalities are present and
        delegates to SentenceTransformer.encode() via the native 'message' modality.
        """
        ds_features = inputs.dataset.features

        active_cols = [
            col
            for col in ("image", "audio", "text")
            if col in ds_features and inputs.dataset[0].get(col) is not None
        ]
        is_text_only = active_cols == ["text"]

        instruction = self._get_instruction(task_metadata, prompt_type)

        all_inputs = [
            {col: batch[col][i] for col in active_cols}
            for batch in inputs
            for i in range(len(batch[active_cols[0]]))
        ]

        # Limit text length if no image/audio is present, otherwise use the model's max context length (32768 tokens).
        # This prevents truncating special tokens in multimodal which raised error while still enabling to limit text-only inputs.
        if is_text_only:
            self.model.max_seq_length = self.max_text_length
        else:
            self.model.max_seq_length = 32768
        return self.model.encode(
            all_inputs,
            prompt=instruction,
            **kwargs,
        )


bidirlm_omni_2_5b = ModelMeta(
    name="BidirLM/BidirLM-Omni-2.5B-Embedding",
    loader=BidirLMOmniEncoder,
    loader_kwargs=dict(
        trust_remote_code=True,
        max_text_length=1024,
    ),
    languages=BIDIRLM_LANGUAGES,
    open_weights=True,
    revision="4d8a7d34095ef8c4ab760744015825b809756dfe",
    release_date="2026-04-07",
    n_parameters=2_445_009_536,
    n_embedding_parameters=315_098_112,
    memory_usage_mb=4663,
    max_tokens=32768,
    embed_dim=POOLING_DIM,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    modalities=["text", "image", "audio"],
    model_type=["dense"],
    reference="https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding",
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/BidirLM/BidirLM-Omni-Contrastive",
    training_datasets=BIDIRLM_OMNI_TRAINING_DATASETS,
    citation=BIDIRLM_CITATION,
)
