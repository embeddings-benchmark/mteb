from typing import Any

import numpy as np
from packaging.version import Version
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import __version__ as transformers_version

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import sentence_transformers_loader
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

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

MODEL_PROMPTS = {
    "Classification": "CLASSIFICATION",
    "MultilabelClassification": "CLASSIFICATION",
    "Clustering": "CLUSTERING",
    "STS": "SIMILARITY",
    PromptType.query.value: "RETRIEVAL_QUERY",
    PromptType.document.value: "RETRIEVAL_DOCUMENT",
}

GECKO_TRAINING_DATA = {
    # Ones that are available from HF.
    "NQHardNegatives",
    "FEVERHardNegatives",
    "HotpotQAHardNegatives",
    "MIRACLRetrievalHardNegatives",
}


class GoogleTextEmbeddingModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        sep: str = " ",
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

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

        max_batch_size = 16  # Vertex API limits the number of instances per call to 250, but there is also a limit of tokens involved. Let's be conservative and set it to 16 by default. TODO: in a future PR, leverage the CountTokens API to get the optimum batch size for each request.
        batches = [
            inputs[i : i + max_batch_size]
            for i in range(0, len(inputs), max_batch_size)
        ]

        all_embeddings = []

        for batch in tqdm(batches, leave=False, disable=not show_progress_bar):
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
        inputs = [text for batch in inputs for text in batch["text"]]

        return self._embed(
            inputs,
            google_task_type=google_task_type,
            show_progress_bar=show_progress_bar,
        )


google_text_emb_004 = ModelMeta(
    loader=GoogleTextEmbeddingModel,  # type: ignore[call-arg]
    loader_kwargs=dict(
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-embedding-004",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,  # assumed
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_text_emb_005 = ModelMeta(
    loader=GoogleTextEmbeddingModel,  # type: ignore[call-arg]
    loader_kwargs=dict(
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-embedding-005",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,  # assumed
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
)

google_text_multilingual_emb_002 = ModelMeta(
    loader=GoogleTextEmbeddingModel,  # type: ignore[call-arg]
    loader_kwargs=dict(
        model_prompts=MODEL_PROMPTS,
    ),
    name="google/text-multilingual-embedding-002",
    model_type=["dense"],
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
    similarity_fn_name=ScoringFunction.COSINE,  # assumed
    framework=["API"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
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
)


def gemma_embedding_loader(model_name: str, revision: str, **kwargs):
    min_transformers_version = "4.56.0"

    if Version(transformers_version) < Version(min_transformers_version):
        raise RuntimeError(
            f"transformers version {transformers_version} is lower than the required "
            f"version {min_transformers_version} to run `{model_name}`"
        )

    return sentence_transformers_loader(model_name, revision, **kwargs)


embedding_gemma_300m = ModelMeta(
    loader=gemma_embedding_loader,
    name="google/embeddinggemma-300m",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
    release_date="2025-09-04",
    n_parameters=307_581_696,
    embed_dim=768,
    max_tokens=2048,
    license="gemma",
    reference="https://ai.google.dev/gemma/docs/embeddinggemma/model_card",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1155,
    citation="""
@misc{vera2025embeddinggemmapowerfullightweighttext,
      title={EmbeddingGemma: Powerful and Lightweight Text Representations},
      author={Henrique Schechter Vera and Sahil Dua and Biao Zhang and Daniel Salz and Ryan Mullins and Sindhu Raghuram Panyam and Sara Smoot and Iftekhar Naim and Joe Zou and Feiyang Chen and Daniel Cer and Alice Lisak and Min Choi and Lucas Gonzalez and Omar Sanseviero and Glenn Cameron and Ian Ballantyne and Kat Black and Kaifeng Chen and Weiyi Wang and Zhe Li and Gus Martins and Jinhyuk Lee and Mark Sherwood and Juyeong Ji and Renjie Wu and Jingxiao Zheng and Jyotinder Singh and Abheesht Sharma and Divyashree Sreepathihalli and Aashi Jain and Adham Elarabawy and AJ Co and Andreas Doumanoglou and Babak Samari and Ben Hora and Brian Potetz and Dahun Kim and Enrique Alfonseca and Fedor Moiseev and Feng Han and Frank Palma Gomez and Gustavo Hernández Ábrego and Hesen Zhang and Hui Hui and Jay Han and Karan Gill and Ke Chen and Koert Chen and Madhuri Shanbhogue and Michael Boratko and Paul Suganthan and Sai Meher Karthik Duddu and Sandeep Mariserla and Setareh Ariafar and Shanfeng Zhang and Shijie Zhang and Simon Baumgartner and Sonam Goenka and Steve Qiu and Tanmaya Dabral and Trevor Walker and Vikram Rao and Waleed Khawaja and Wenlei Zhou and Xiaoqi Ren and Ye Xia and Yichang Chen and Yi-Ting Chen and Zhe Dong and Zhongli Ding and Francesco Visin and Gaël Liu and Jiageng Zhang and Kathleen Kenealy and Michelle Casbon and Ravin Kumar and Thomas Mesnard and Zach Gleicher and Cormac Brick and Olivier Lacombe and Adam Roberts and Qin Yin and Yunhsuan Sung and Raphael Hoffmann and Tris Warkentin and Armand Joulin and Tom Duerig and Mojtaba Seyedhosseini},
      year={2025},
      eprint={2509.20354},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.20354},
}""",
)
