from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import tqdm

from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

MULTILINGUAL_EVALUATED_LANGUAGES = [
    "arb_Arab",
    "ben_Beng",
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "pes_Arab",
    "fin_Latn",
    "fra_Latn",
    "hin_Deva",
    "ind_Latn",
    "jpn_Jpan",
    "kor_Hang",
    "rus_Cyrl",
    "swh_Latn",
    "tel_Telu",
    "tha_Thai",
    "yor_Latn",
    "zho_Hant",
    "zho_Hans",
]

MODEL_PROMPTS = {
    "Classification": "CLASSIFICATION",
    "MultilabelClassification": "CLASSIFICATION",
    "Clustering": "CLUSTERING",
    "STS": "SIMILARITY",
    PromptType.query.value: "RETRIEVAL_QUERY",
    PromptType.passage.value: "RETRIEVAL_DOCUMENT",
}

GECKO_TRAINING_DATA = {
    # Ones that are available from HF.
    "NQHardNegatives": ["train"],
    "FEVERHardNegatives": ["train"],
    "HotpotQAHardNegatives": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
}


class GoogleTextEmbeddingModel(Encoder, Wrapper):
    def __init__(
        self,
        model_name: str,
        sep: str = " ",
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_prompts = (
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def _embed(
        self,
        texts: list[str],
        google_task_type: str | None = None,
        show_progress_bar: bool = False,
        titles: list[str] | None = None,
        dimensionality: int | None = 768,
    ) -> list[list[float]]:
        """Embeds texts with a pre-trained, foundational model.
        From https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#generative-ai-get-text-embedding-python_vertex_ai_sdk
        """
        requires_package(
            self, "vertexai", self.model_name, "pip install 'mteb[vertexai]'"
        )
        from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained(self.model_name)
        if titles:
            # Allow title-only embeddings by replacing text with a space
            # Else Google throws google.api_core.exceptions.InvalidArgument: 400 The text content is empty.
            inputs = [
                TextEmbeddingInput(
                    text if text else " ", task_type=google_task_type, title=title
                )
                for text, title in zip(texts, titles)
            ]
        else:
            inputs = [
                TextEmbeddingInput(text, task_type=google_task_type) for text in texts
            ]

        kwargs = {"output_dimensionality": dimensionality} if dimensionality else {}

        max_batch_size = 16  ## Vertex API limits the number of instances per call to 250, but there is also a limit of tokens involved. Let's be conservative and set it to 16 by default. TODO: in a future PR, leverage the CountTokens API to get the optimum batch size for each request.
        batches = [
            inputs[i : i + max_batch_size]
            for i in range(0, len(inputs), max_batch_size)
        ]

        all_embeddings = []

        for batch in tqdm.tqdm(batches, leave=False, disable=not show_progress_bar):
            try:
                embeddings_batch = model.get_embeddings(batch, **kwargs)
            # Except the very rare google.api_core.exceptions.InternalServerError
            except Exception as e:
                print("Retrying once after error:", e)
                embeddings_batch = model.get_embeddings(batch, **kwargs)

            all_embeddings.extend([embedding.values for embedding in embeddings_batch])

        return np.asarray(all_embeddings)

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        prompt_name = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        google_task_type = self.model_prompts.get(prompt_name)

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )

        return self._embed(
            sentences,
            google_task_type=google_task_type,
            show_progress_bar=show_progress_bar,
        )


google_text_emb_004 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="text-embedding-004",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-embedding-004",
    languages=["eng-Latn"],
    open_weights=False,
    revision="1",  # revision is intended for implementation
    release_date="2024-05-14",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_text_emb_005 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="text-embedding-005",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-embedding-005",
    languages=["eng-Latn"],
    open_weights=False,
    revision="1",  # revision is intended for implementation
    release_date="2024-11-18",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_text_multilingual_emb_002 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="text-multilingual-embedding-002",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-multilingual-embedding-002",
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,  # From the list of evaluated languages in https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#supported_text_languages
    open_weights=False,
    revision="1",
    release_date="2024-05-14",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=768,
    license=None,
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_gemini_embedding_exp_03_07 = ModelMeta(
    loader=partial(
        GoogleTextEmbeddingModel,
        model_name="gemini-embedding-exp-03-07",
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/gemini-embedding-exp-03-07",
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=False,
    revision="1",
    release_date="2025-03-07",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=3072,
    license=None,
    reference="https://developers.googleblog.com/en/gemini-embedding-text-model-now-available-gemini-api/",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)
