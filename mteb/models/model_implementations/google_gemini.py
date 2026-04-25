from __future__ import annotations

import asyncio
import io
import logging
import random
import re
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.google_text_embedding import (
    GECKO_TRAINING_DATA,
    MODEL_PROMPTS,
    GoogleTextEmbeddingModel,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

MULTILINGUAL_EVALUATED_LANGUAGES = [
    "arb-Arab",
    "ben-Beng",
    "eng-Latn",
    "spa-Latn",
    "deu-Latn",
    "pes-Arab",
    "fin-Latn",
    "fra-Latn",
    "hin-Deva",
    "ind-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "rus-Cyrl",
    "swh-Latn",
    "tel-Telu",
    "tha-Thai",
    "yor-Latn",
    "zho-Hant",
    "zho-Hans",
]


# Prompt mapping for Gemini Embedding 2.
# Task-type defaults are derived from MTEB metadata; per-task overrides
# are from Google's recommended mapping in issue #4260.
GEMINI_EMBEDDING_2_PROMPTS = {
    # Task-type defaults (derived from metadata)
    "Classification": "CLASSIFICATION",
    "MultilabelClassification": "CLASSIFICATION",
    "Clustering": "CLUSTERING",
    "STS": "SEMANTIC_SIMILARITY",
    "PairClassification": "CLASSIFICATION",
    "BitextMining": "SEMANTIC_SIMILARITY",
    "Reranking": "RETRIEVAL_DOCUMENT",
    "Summarization": "SEMANTIC_SIMILARITY",
    "InstructionRetrieval": "RETRIEVAL_DOCUMENT",
    "InstructionReranking": "RETRIEVAL_DOCUMENT",
    PromptType.query.value: "RETRIEVAL_QUERY",
    PromptType.document.value: "RETRIEVAL_DOCUMENT",
    # Per-task overrides from Google's recommended mapping (issue #4260).
    # Only tasks where Google's recommendation differs from the task-type default.
    "ArXivHierarchicalClusteringP2P": "RETRIEVAL_DOCUMENT",
    "ArguAna": "FACT_VERIFICATION",
    "ArmenianParaphrasePC": "FACT_VERIFICATION",
    "BUCC.v2": "QUESTION_ANSWERING",
    "BiorxivClusteringP2P.v2": "CLASSIFICATION",
    "CataloniaTweetClassification": "CLUSTERING",
    "Core17InstructionRetrieval": "QUESTION_ANSWERING",
    "CyrillicTurkicLangClassification": "CLUSTERING",
    "DalajClassification": "FACT_VERIFICATION",
    "FinParaSTS": "RETRIEVAL_DOCUMENT",
    "FinancialPhrasebankClassification": "SEMANTIC_SIMILARITY",
    "GreekLegalCodeClassification": "CLUSTERING",
    "GujaratiNewsClassification": "CLUSTERING",
    "HagridRetrieval": "SEMANTIC_SIMILARITY",
    "IndicCrosslingualSTS": "RETRIEVAL_DOCUMENT",
    "IndicLangClassification": "CLUSTERING",
    "IsiZuluNewsClassification": "QUESTION_ANSWERING",
    "JSICK": "FACT_VERIFICATION",
    "LEMBPasskeyRetrieval": "CLUSTERING",
    "MLQARetrieval": "QUESTION_ANSWERING",
    "MalteseNewsClassification": "CLUSTERING",
    "MasakhaNEWSClassification": "CLUSTERING",
    "MultiEURLEXMultilabelClassification": "CLUSTERING",
    "NepaliNewsClassification": "CLUSTERING",
    "News21InstructionRetrieval": "CLASSIFICATION",
    "NollySentiBitextMining": "FACT_VERIFICATION",
    "NordicLangClassification": "CLUSTERING",
    "NorwegianCourtsBitextMining": "FACT_VERIFICATION",
    "NusaTranslationBitextMining": "QUESTION_ANSWERING",
    "NusaXBitextMining": "QUESTION_ANSWERING",
    "OdiaNewsClassification": "CLUSTERING",
    "OpusparcusPC": "SEMANTIC_SIMILARITY",
    "PAC": "FACT_VERIFICATION",
    "PawsXPairClassification": "FACT_VERIFICATION",
    "PoemSentimentClassification": "SEMANTIC_SIMILARITY",
    "PunjabiNewsClassification": "FACT_VERIFICATION",
    "Robust04InstructionRetrieval": "QUESTION_ANSWERING",
    "RomaniBibleClustering": "SEMANTIC_SIMILARITY",
    "RuBQReranking": "CLASSIFICATION",
    "SCIDOCS": "FACT_VERIFICATION",
    "STSES": "QUESTION_ANSWERING",
    "ScalaClassification": "FACT_VERIFICATION",
    "SemRel24STS": "QUESTION_ANSWERING",
    "SinhalaNewsClassification": "CLUSTERING",
    "SiswatiNewsClassification": "QUESTION_ANSWERING",
    "SpartQA": "SEMANTIC_SIMILARITY",
    "SprintDuplicateQuestions": "SEMANTIC_SIMILARITY",
    "StackOverflowQA": "QUESTION_ANSWERING",
    "SwahiliNewsClassification": "CLUSTERING",
    "SwissJudgementClassification": "RETRIEVAL_DOCUMENT",
    "T2Reranking": "CLASSIFICATION",
    "TRECCOVID": "FACT_VERIFICATION",
    "TempReasonL1": "SEMANTIC_SIMILARITY",
    "TswanaNewsClassification": "CLUSTERING",
    "TweetTopicSingleClassification": "CLUSTERING",
    "TwitterURLCorpus": "SEMANTIC_SIMILARITY",
    "VoyageMMarcoReranking": "SEMANTIC_SIMILARITY",
    "WebLINXCandidatesReranking": "CLASSIFICATION",
    "WikipediaRetrievalMultilingual": "FACT_VERIFICATION",
    "WinoGrande": "QUESTION_ANSWERING",
    "XNLI": "SEMANTIC_SIMILARITY",
}


def _audio_to_wav_bytes(audio_item: dict) -> bytes:
    """Convert an AudioInputItem (numpy array + sampling_rate) to WAV bytes."""

    array = audio_item["array"]
    sampling_rate = audio_item["sampling_rate"]
    # Convert float audio to 16-bit PCM
    pcm = (array * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class GoogleGeminiEmbeddingModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        model_prompts: dict[str, str] | None = None,
        embed_dim: int | None = None,
        **kwargs,
    ) -> None:
        from google import genai

        # Strip org prefix — the Gemini API expects bare model names
        self.model_name = model_name.split("/", 1)[-1]
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)
        self.embed_dim = embed_dim
        self.client = genai.Client()

    def _embed(
        self,
        contents: list,
        google_task_type: str | None = None,
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        from google.genai.types import EmbedContentConfig

        config = EmbedContentConfig(
            taskType=google_task_type,
            outputDimensionality=self.embed_dim,
        )

        async def run() -> list:
            semaphore = asyncio.Semaphore(batch_size)

            async def embed_one(item: Any) -> list[float]:
                wait_time = 1.0
                async with semaphore:
                    for attempt in range(10):
                        try:
                            response = await self.client.aio.models.embed_content(
                                model=self.model_name, contents=item, config=config
                            )
                            return response.embeddings[0].values
                        except Exception as e:
                            if "429" in str(e) and attempt < 9:
                                match = re.search(
                                    r"retry in (\d+(?:\.\d+)?)s", str(e), re.IGNORECASE
                                )
                                wait = float(match.group(1)) if match else wait_time
                                wait += random.uniform(0, 5)
                                logger.warning(
                                    f"Rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/10)"
                                )
                                await asyncio.sleep(wait)
                                wait_time = min(wait_time * 2, 60)
                            elif attempt < 9:
                                logger.warning(
                                    f"Retrying after error (attempt {attempt + 1}/10): {e}"
                                )
                            else:
                                raise

            if show_progress_bar:
                from tqdm.asyncio import tqdm as async_tqdm

                return await async_tqdm.gather(
                    *[embed_one(item) for item in contents], leave=False
                )
            return await asyncio.gather(*[embed_one(item) for item in contents])

        # run in a separate thread to avoid blocking if the event loop is already running (e.g. in a notebook)
        with ThreadPoolExecutor(max_workers=1) as pool:
            embeddings = pool.submit(asyncio.run, run()).result()

        return np.asarray(embeddings)

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
        from google.genai.types import Part

        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        google_task_type = self.model_prompts.get(prompt_name)

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )
        batch_size = kwargs.pop("batch_size", 32)

        has_text = "text" in inputs.dataset.features
        has_image = "image" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features
        has_title = "title" in inputs.dataset.features

        if has_text and has_image:
            contents = []
            for batch in inputs:
                for text, image in zip(batch["text"], batch["image"]):
                    contents.append([text, image])
        elif has_text and has_audio:
            contents = []
            for batch in inputs:
                for text, audio_item in zip(batch["text"], batch["audio"]):
                    wav_bytes = _audio_to_wav_bytes(audio_item)
                    contents.append(
                        [text, Part.from_bytes(data=wav_bytes, mime_type="audio/wav")]
                    )
        elif has_text:
            if has_title:
                contents = []
                for batch in inputs:
                    for title, text in zip(batch["title"], batch["text"]):
                        if title:
                            contents.append(f"title: {title} | text: {text}")
                        else:
                            contents.append(text)
            else:
                contents = [text for batch in inputs for text in batch["text"]]
        elif has_image:
            contents = [img for batch in inputs for img in batch["image"]]
        elif has_audio:
            contents = []
            for batch in inputs:
                for audio_item in batch["audio"]:
                    wav_bytes = _audio_to_wav_bytes(audio_item)
                    contents.append(
                        Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
                    )
        else:
            raise ValueError("No text, image, or audio features found in inputs")

        return self._embed(
            contents,
            google_task_type=google_task_type,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )


google_gemini_embedding_001 = ModelMeta(
    loader=GoogleTextEmbeddingModel,  # type: ignore[call-arg]
    loader_kwargs=dict(
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/gemini-embedding-001",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=False,
    revision="1",
    release_date="2025-03-07",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=3072,
    license=None,
    reference="https://ai.google.dev/gemini-api/docs/embeddings",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    extra_requirements_groups=["vertexai"],
)

google_gemini_embedding_2_preview = ModelMeta(
    loader=GoogleGeminiEmbeddingModel,  # type: ignore[call-arg]
    loader_kwargs=dict(
        model_prompts=GEMINI_EMBEDDING_2_PROMPTS,
        embed_dim=3072,
    ),
    name="google/gemini-embedding-2-preview",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    modalities=["audio", "image", "text"],
    open_weights=False,
    revision="1",
    release_date="2025-03-25",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=2048,
    embed_dim=[768, 1536, 3072],
    license=None,
    reference="https://ai.google.dev/gemini-api/docs/embeddings",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    extra_requirements_groups=["google_genai"],
)
