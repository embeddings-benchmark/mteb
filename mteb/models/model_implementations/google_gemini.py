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
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_implementations.google_text_embedding import (
    GECKO_TRAINING_DATA,
    MODEL_PROMPTS,
    GoogleTextEmbeddingModel,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch
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

GEMINI_VIDEO_FPS = 1.0
GEMINI_MAX_VIDEO_FRAMES = 32
GEMINI_MAX_AUDIO_SECONDS = 180

GEMINI_EMBEDDING_CITATION = """@misc{lee2025geminiembeddinggeneralizableembeddings,
  title={Gemini Embedding: Generalizable Embeddings from Gemini},
  author={Jinhyuk Lee and Feiyang Chen and Sahil Dua and Daniel Cer and Madhuri Shanbhogue and Iftekhar Naim and Gustavo Hernández Ábrego and Zhe Li and Kaifeng Chen and Henrique Schechter Vera and Xiaoqi Ren and Shanfeng Zhang and Daniel Salz and Michael Boratko and Jay Han and Blair Chen and Shuo Huang and Vikram Rao and Paul Suganthan and Feng Han and Andreas Doumanoglou and Nithi Gupta and Fedor Moiseev and Cathy Yip and Aashi Jain and Simon Baumgartner and Shahrokh Shahi and Frank Palma Gomez and Sandeep Mariserla and Min Choi and Parashar Shah and Sonam Goenka and Ke Chen and Ye Xia and Koert Chen and Sai Meher Karthik Duddu and Yichang Chen and Trevor Walker and Wenlei Zhou and Rakesh Ghiya and Zach Gleicher and Karan Gill and Zhe Dong and Mojtaba Seyedhosseini and Yunhsuan Sung and Raphael Hoffmann and Tom Duerig},
  year={2025},
  eprint={2503.07891},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2503.07891},
}"""

GEMINI_EMBEDDING_2_CITATION = """@misc{shanbhogue2026geminiembedding2native,
  title={Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini},
  author={Madhuri Shanbhogue and Zhe Li and Shanfeng Zhang and Gustavo Hernández Ábrego and Shih-Cheng Huang and Aashi Jain and Daniel Salz and Sonam Goenka and Chaitra Hegde and Ji Ma and Feiyang Chen and Jiaxing Wu and Tanmaya Dabral and Babak Samari and Kevin Poulet and Daniel Cer and Kaifeng Chen and Paul Suganathan and Hui Hui and Jovan Andonov and Philippe Schlattner and Jay Han and Iftekhar Naim and Wing Lowe and Vladimir Pchelin and Albert Yang and Yi-Ting Chen and Zhongli Ding and Grace Zhang and Georg Heigold and Yichang Chen and Antoine Reveillon and Brendan Mccloskey and Wenlei Zhou and Dahun Kim and Rui Meng and Emma Wang and Jack Zheng and Halley Fede and Zhen Yang and Keegan Mosley and Brian Potetz and Sahil Dua and Henrique Schechter Vera and Shen Gao and Hesen Zhang and Andreas Hess and Hengxuan Ying and Alberto Montes and Karan Gill and Min Choi and Sebastian Russo and Anja Hauth and Jinhyuk Lee and Michael Boratko and Megan Barnes and Vikram Rao and Claudiu Musat and Cyril Allauzen and Ehsan Variani and Shankar Kumar and Tom Bagby and Junyi Jiao and Yang Gu and Tengxin Li and Ayush Agrawal and Roberto Santana and Dev Nath and Stephen Karukas and Shuoxuan Han and Lucia Loher and Alice Twu and Nidhi Vyas and Siddharth Bhai and Frank Palma Gomez and Wangyuan Zhang and Chaoren Liu and Jizheng Yang and Steve Qiu and Shijie Zhang and Sujay Kulkarni and Sascha Rothe and Sean Nakamoto and Raphael Hoffmann and Zach Gleicher and Yunhsuan Sung and Qin Yin and Tom Duerig and Mojtaba Seyedhosseini},
  year={2026},
  eprint={2605.27295},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.27295},
}"""


# Prompt mapping for Gemini Embedding 2.
# Task-type defaults are derived from MTEB metadata; per-task overrides
# are from Google's recommended mapping in issue #4260.
GEMINI_EMBEDDING_2_TEXT_PREFIXES = {
    "RETRIEVAL_QUERY": "task: search result | query: {text}",
    "QUESTION_ANSWERING": "task: question answering | query: {text}",
    "FACT_VERIFICATION": "task: fact checking | query: {text}",
    "CODE_RETRIEVAL_QUERY": "task: code retrieval | query: {text}",
    "CLASSIFICATION": "task: classification | query: {text}",
    "CLUSTERING": "task: clustering | query: {text}",
    "SEMANTIC_SIMILARITY": "task: sentence similarity | query: {text}",
}


def _format_gemini_embedding_2_document(text: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {text}"


def _format_gemini_embedding_2_text(
    text: str,
    google_task_type: str | None,
    prompt_type: PromptType | None,
    title: str | None = None,
) -> str:
    if prompt_type == PromptType.document:
        return _format_gemini_embedding_2_document(text, title)
    if google_task_type in GEMINI_EMBEDDING_2_TEXT_PREFIXES:
        return GEMINI_EMBEDDING_2_TEXT_PREFIXES[google_task_type].format(text=text)
    return text


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

    # Preserve the original sampling rate and truncate to Gemini's duration limit.
    num_samples = array.shape[-1]
    max_samples = sampling_rate * GEMINI_MAX_AUDIO_SECONDS
    if num_samples > max_samples:
        logger.warning(
            "Truncating Gemini Embedding 2 audio input from %.2f to %d seconds.",
            num_samples / sampling_rate,
            GEMINI_MAX_AUDIO_SECONDS,
        )
        array = array[..., :max_samples]

    # Convert float audio to 16-bit PCM
    pcm = (array * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _video_to_mp4_bytes(frames: torch.Tensor, *, fps: float) -> bytes:
    """Encode sampled ``[T, C, H, W]`` RGB frames as an in-memory MP4."""
    import torch
    from torch.nn import functional as F
    from torchcodec.encoders import VideoEncoder

    if frames.ndim != 4:
        raise ValueError(
            f"Expected video frames with shape [T, C, H, W], got {tuple(frames.shape)}"
        )
    if frames.shape[0] == 0:
        raise ValueError("Expected at least one video frame")
    if frames.shape[1] != 3:
        raise ValueError(
            f"Expected RGB video frames with 3 channels, got {frames.shape[1]}"
        )
    if fps <= 0:
        raise ValueError(f"Expected a positive video FPS, got {fps}")

    video_frames = frames.detach().cpu()
    if video_frames.is_floating_point():
        # TorchCodec currently returns uint8, but accepting normalized tensors
        # makes this boundary safe for other frame providers too.
        if video_frames.numel():
            min_value = video_frames.min().item()
            max_value = video_frames.max().item()

            if min_value >= 0 and max_value <= 1:
                video_frames = torch.mul(video_frames, 255)
    if video_frames.dtype != torch.uint8:
        video_frames = video_frames.clamp(0, 255).to(torch.uint8)

    height, width = video_frames.shape[2:4]
    # H.264 with yuv420p requires even dimensions.
    if height == 0 or width == 0:
        raise ValueError("Video frames are too small to encode as MP4")
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        video_frames = F.pad(
            video_frames,
            (0, pad_w, 0, pad_h),
            mode="replicate",
        )

    video_frames = video_frames.contiguous()
    encoded = VideoEncoder(video_frames, frame_rate=fps).to_tensor(
        "mp4",
        codec="h264",
        pixel_format="yuv420p",
    )
    return encoded.numpy().tobytes()


def _build_gemini_content(
    *,
    text: str | None,
    title: str | None,
    image: Any | None,
    audio: dict | None,
    video: torch.Tensor | None,
    google_task_type: str | None,
    prompt_type: PromptType | None,
    use_text_formatting: bool = True,
) -> str | list[Any] | Any:
    """Build one Gemini input, aggregating all modalities present in a row."""
    from google.genai.types import Part

    parts: list[Any] = []
    has_media = image is not None or audio is not None or video is not None
    if text is not None:
        formatted_text = (
            text
            if has_media or not use_text_formatting
            else _format_gemini_embedding_2_text(
                text, google_task_type, prompt_type, title
            )
        )
        parts.append(formatted_text)
    if image is not None:
        parts.append(image)
    if audio is not None:
        parts.append(
            Part.from_bytes(
                data=_audio_to_wav_bytes(audio),
                mime_type="audio/wav",
            )
        )
    if video is not None:
        parts.append(
            Part.from_bytes(
                data=_video_to_mp4_bytes(video, fps=GEMINI_VIDEO_FPS),
                mime_type="video/mp4",
            )
        )

    if not parts:
        raise ValueError("No supported Gemini input modality found")
    return parts[0] if len(parts) == 1 else parts


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
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        from google.genai.types import EmbedContentConfig

        config = EmbedContentConfig(outputDimensionality=self.embed_dim)

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
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        google_task_type = self.model_prompts.get(prompt_name)

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )
        batch_size = kwargs.pop("batch_size", 32)

        features = inputs.dataset.features
        # Apply prompts only to text-only tasks: https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
        use_text_formatting = set(task_metadata.modalities) == {"text"}
        if (
            use_text_formatting
            and prompt_type == PromptType.document
            and "body" in features
        ):
            text_key = "body"
        else:
            text_key = "text"

        has_video = "video" in features
        if has_video:
            # Sample at 1 FPS and cap at 32 frames to match Gemini's video limits.
            inputs.collate_fn = FramesCollator(
                fps=GEMINI_VIDEO_FPS,
                max_frames=GEMINI_MAX_VIDEO_FRAMES,
            )

        modality_keys = (text_key, "title", "image", "audio", "video")
        contents = []
        for batch in inputs:
            num_samples = len(next(iter(batch.values())))
            for index in range(num_samples):
                row = {key: batch[key][index] for key in modality_keys if key in batch}
                contents.append(
                    _build_gemini_content(
                        text=row.get(text_key),
                        title=row.get("title"),
                        image=row.get("image"),
                        audio=row.get("audio"),
                        video=row.get("video"),
                        google_task_type=google_task_type,
                        prompt_type=prompt_type,
                        use_text_formatting=use_text_formatting,
                    )
                )

        return self._embed(
            contents,
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
    release_date="2025-07-14",
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
    citation=GEMINI_EMBEDDING_CITATION,
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
    modalities=["audio", "image", "text", "video"],
    open_weights=False,
    revision="1",
    release_date="2026-03-10",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=[768, 1536, 3072],
    license=None,
    reference="https://ai.google.dev/gemini-api/docs/embeddings",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    superseded_by="google/gemini-embedding-2",
    extra_requirements_groups=["google_genai"],
    citation=GEMINI_EMBEDDING_2_CITATION,
)

google_gemini_embedding_2 = ModelMeta(
    loader=GoogleGeminiEmbeddingModel,  # type: ignore[call-arg]
    loader_kwargs=dict(
        model_prompts=GEMINI_EMBEDDING_2_PROMPTS,
        embed_dim=3072,
    ),
    name="google/gemini-embedding-2",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    modalities=["audio", "image", "text", "video"],
    open_weights=False,
    revision="1",
    release_date="2026-04-22",
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=[768, 1536, 3072],
    license=None,
    reference="https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    extra_requirements_groups=["google_genai"],
    citation=GEMINI_EMBEDDING_2_CITATION,
)
