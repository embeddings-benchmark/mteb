import pytest

import mteb
from mteb.models.model_meta import ModelMeta

# Historic models with n_embedding_parameters=None. Do NOT add new models to this list.
_MISSING_N_EMBEDDING_MODELS = [
    "ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-3B-v1",
    "ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1",
    "BAAI/bge-visualized-base",
    "BAAI/bge-visualized-m3",
    "ByteDance/ListConRanker",
    "Bytedance/Seed1.6-embedding",
    "GritLM/GritLM-8x7B",
    "Kingsoft-LLM/QZhou-Embedding-Zh",
    "NovaSearch/stella_en_400M_v5",
    "QuanSun/EVA02-CLIP-B-16",
    "QuanSun/EVA02-CLIP-L-14",
    "QuanSun/EVA02-CLIP-bigE-14",
    "QuanSun/EVA02-CLIP-bigE-14-plus",
    "Salesforce/SFR-Embedding-Code-2B_R",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-6.7b-coco",
    "Snowflake/snowflake-arctic-embed-m-long",
    "Snowflake/snowflake-arctic-embed-m-v2.0",
    "TIGER-Lab/VLM2Vec-Full",
    "TIGER-Lab/VLM2Vec-LoRA",
    "TencentBAC/Conan-embedding-v2",
    "TomoroAI/tomoro-colqwen3-embed-4b",
    "TomoroAI/tomoro-colqwen3-embed-8b",
    "VPLabs/SearchMap_Preview",
    "ai-sage/Giga-Embeddings-instruct",
    "baseline/bb25",
    "baseline/bm25s",
    "codefuse-ai/C2LLM-0.5B",
    "codefuse-ai/C2LLM-7B",
    "consciousAI/cai-stellaris-text-embeddings",
    "deepvk/USER2-base",
    "deepvk/USER2-small",
    "dmedhi/PawanEmbd-68M",
    "facebook/SONAR",
    "facebook/dinov2-base",
    "facebook/dinov2-giant",
    "facebook/dinov2-large",
    "facebook/dinov2-small",
    "facebook/webssl-dino1b-full2b-224",
    "facebook/webssl-dino2b-full2b-224",
    "facebook/webssl-dino2b-heavy2b-224",
    "facebook/webssl-dino2b-light2b-224",
    "facebook/webssl-dino300m-full2b-224",
    "facebook/webssl-dino3b-full2b-224",
    "facebook/webssl-dino3b-heavy2b-224",
    "facebook/webssl-dino3b-light2b-224",
    "facebook/webssl-mae1b-full2b-224",
    "facebook/webssl-mae300m-full2b-224",
    "ibm-granite/granite-vision-3.3-2b-embedding",
    "infly/inf-retriever-v1",
    "intfloat/mmE5-mllama-11b-instruct",
    "jinaai/jina-clip-v1",
    "jinaai/jina-colbert-v2",
    "jinaai/jina-reranker-v2-base-multilingual",
    "jxm/cde-small-v1",
    "jxm/cde-small-v2",
    "kakaobrain/align-base",
    "laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    "lightonai/GTE-ModernColBERT-v1",
    "malenia1/ternary-weight-embedding",
    "microsoft/LLM2CLIP-Openai-B-16",
    "microsoft/LLM2CLIP-Openai-L-14-224",
    "microsoft/LLM2CLIP-Openai-L-14-336",
    "mixedbread-ai/mxbai-edge-colbert-v0-17m",
    "mixedbread-ai/mxbai-edge-colbert-v0-32m",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "mixedbread-ai/mxbai-rerank-large-v1",
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "nomic-ai/colnomic-embed-multimodal-3b",
    "nomic-ai/colnomic-embed-multimodal-7b",
    "nomic-ai/nomic-embed-code",
    "nomic-ai/nomic-embed-text-v1",
    "nomic-ai/nomic-embed-text-v1-ablated",
    "nomic-ai/nomic-embed-text-v1-unsupervised",
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-vision-v1.5",
    "nvidia/NV-Embed-v1",
    "nvidia/NV-Embed-v2",
    "nvidia/llama-nemoretriever-colembed-1b-v1",
    "nvidia/llama-nemoretriever-colembed-3b-v1",
    "nyu-visionx/moco-v3-vit-b",
    "nyu-visionx/moco-v3-vit-l",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "samaya-ai/RepLLaMA-reproduced",
    "samaya-ai/promptriever-llama2-7b-v1",
    "samaya-ai/promptriever-llama3.1-8b-instruct-v1",
    "samaya-ai/promptriever-llama3.1-8b-v1",
    "sensenova/piccolo-large-zh-v2",
    "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "tencent/Youtu-Embedding",
    "vidore/colSmol-256M",
    "vidore/colSmol-500M",
    "vidore/colpali-v1.1",
    "vidore/colpali-v1.2",
    "vidore/colpali-v1.3",
    "vidore/colqwen2-v1.0",
    "vidore/colqwen2.5-v0.2",
    "voyageai/voyage-multimodal-3",
    "facebook/webssl-dino5b-full2b-224",
    "facebook/webssl-dino7b-full8b-224",
    "facebook/webssl-dino7b-full8b-378",
    "facebook/webssl-dino7b-full8b-518",
    "facebook/webssl-mae700m-full2b-224",
    "OrlikB/KartonBERT-USE-base-v1",
    "OrlikB/st-polish-kartonberta-base-alpha-v1",
    "OpenSearch-AI/Ops-Colqwen3-4B",
    "jinaai/jina-clip-v2",
    "Bytedance/Seed1.6-embedding-1215",
    "Cohere/Cohere-embed-v4.0",
    "Cohere/Cohere-embed-v4.0 (output_dtype=binary)",
    "Cohere/Cohere-embed-v4.0 (output_dtype=int8)",
    "cohere/embed-english-v3.0",
    "cohere/embed-multilingual-v3.0",
    "jinaai/jina-embeddings-v3",
    "jinaai/jina-embeddings-v4",
    "voyageai/voyage-2",
    "voyageai/voyage-3",
    "voyageai/voyage-3.5",
    "voyageai/voyage-3.5 (output_dtype=binary)",
    "voyageai/voyage-3.5 (output_dtype=int8)",
    "voyageai/voyage-3-m-exp",
    "voyageai/voyage-3-large",
    "voyageai/voyage-3-lite",
    "voyageai/voyage-4",
    "voyageai/voyage-4-large",
    "voyageai/voyage-4-large (embed_dim=2048)",
    "voyageai/voyage-4-lite",
    "voyageai/voyage-code-2",
    "voyageai/voyage-code-3",
    "voyageai/voyage-finance-2",
    "voyageai/voyage-large-2",
    "voyageai/voyage-large-2-instruct",
    "voyageai/voyage-law-2",
    "voyageai/voyage-multilingual-2",
    "bedrock/amazon-titan-embed-text-v1",
    "bedrock/amazon-titan-embed-text-v2",
    "bedrock/cohere-embed-english-v3",
    "bedrock/cohere-embed-multilingual-v3",
    "Cohere/Cohere-embed-english-v3.0",
    "Cohere/Cohere-embed-english-light-v3.0",
    "Cohere/Cohere-embed-multilingual-v3.0",
    "Cohere/Cohere-embed-multilingual-light-v3.0",
    "ByteDance-Seed/Seed1.5-Embedding",
    "Bytedance/Seed1.6-embedding-1215",
    "baseline/Human",
    "baseline/random-cross-encoder-baseline",
    "baseline/random-encoder-baseline",
    "amazon/Titan-text-embeddings-v2",
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-large (embed_dim=512)",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-small (embed_dim=512)",
    "openai/text-embedding-ada-002",
    "nvidia/llama-nemotron-rerank-1b-v2",
    "nvidia/llama-nemotron-colembed-vl-3b-v2",
    "nvidia/nemotron-colembed-vl-4b-v2",
    "nvidia/nemotron-colembed-vl-8b-v2",
    "VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1",
    "VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1",
    "VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1",
    "VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1",
    "VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1",
    "VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1",
    "MCINext/Hakim",
    "MCINext/Hakim-small",
    "MCINext/Hakim-unsup",
    "google/gemini-embedding-001",
    "google/text-embedding-004",
    "google/text-embedding-005",
    "google/text-multilingual-embedding-002",
    "ICT-TIME-and-Querit/BOOM_4B_v1",
    # audio models
    "google/vggish",
    "LCO-Embedding/LCO-Embedding-Omni-3B",
    "LCO-Embedding/LCO-Embedding-Omni-7B",
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large",
    "facebook/wav2vec2-large-xlsr-53",
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "facebook/wav2vec2-xls-r-1b",
    "facebook/wav2vec2-xls-r-2b",
    "facebook/wav2vec2-xls-r-2b-21-to-en",
    "facebook/wav2vec2-xls-r-300m",
    "vitouphy/wav2vec2-xls-r-300m-phoneme",
    "laion/clap-htsat-fused",
    "laion/clap-htsat-unfused",
    "laion/larger_clap_general",
    "laion/larger_clap_music",
    "laion/larger_clap_music_and_speech",
    "lyrebird/wav2clip",
    "microsoft/msclap-2022",
    "microsoft/msclap-2023",
    "microsoft/unispeech-sat-base-100h-libri-ft",
    "microsoft/wavlm-base",
    "microsoft/wavlm-base-plus",
    "microsoft/wavlm-base-plus-sd",
    "microsoft/wavlm-base-plus-sv",
    "microsoft/wavlm-base-sd",
    "microsoft/wavlm-base-sv",
    "microsoft/wavlm-large",
    "facebook/seamless-m4t-v2-large",
    "Qwen/Qwen2-Audio-7B",
    "facebook/hubert-base-ls960",
    "facebook/hubert-large-ls960-ft",
    "baseline/random-cross-encoder-baseline",
    "baseline/random-encoder-baseline",
    "facebook/encodec_24khz",
    "speechbrain/m-ctc-t-large",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "asapp/sew-d-base-plus-400k-ft-ls100h",
    "asapp/sew-d-mid-400k-ft-ls100h",
    "asapp/sew-d-tiny-100k-ft-ls100h",
    "google/yamnet",
    "openai/whisper-base",
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-tiny",
    "microsoft/speecht5_asr",
    "microsoft/speecht5_multimodal",
    "facebook/mms-1b-all",
    "facebook/mms-1b-fl102",
    "facebook/mms-1b-l1107",
    "facebook/data2vec-audio-base-960h",
    "facebook/data2vec-audio-large-960h",
    "OpenMuQ/MuQ-MuLan-large",
    "speechbrain/cnn14-esc50",
    "microsoft/speecht5_tts",
]


@pytest.mark.parametrize(
    "training_datasets",
    [
        {"Touche2020"},  # parent task
        {"Touche2020-NL"},  # child task
    ],
)
def test_model_similar_tasks(training_datasets):
    dummy_model_meta = ModelMeta(
        name="test/test_model",
        revision="test",
        release_date=None,
        languages=None,
        loader=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=None,
        license=None,
        open_weights=None,
        public_training_code=None,
        public_training_data=None,
        framework=[],
        reference=None,
        similarity_fn_name=None,
        use_instructions=None,
        training_datasets=training_datasets,
        adapted_from=None,
        superseded_by=None,
    )
    expected = sorted(
        [
            "NanoTouche2020Retrieval",
            "Touche2020",
            "Touche2020-Fa",
            "Touche2020-Fa.v2",
            "Touche2020-NL",
            "Touche2020-VN",
            "Touche2020-PL",
            "Touche2020Retrieval.v3",
        ]
    )
    assert sorted(dummy_model_meta.get_training_datasets()) == expected


def test_similar_tasks_superseded_by():
    """Banking77Classification in model training data, but Banking77Classification.v2 version not"""
    model_meta = mteb.get_model_meta("BAAI/bge-multilingual-gemma2")
    assert "Banking77Classification.v2" in model_meta.get_training_datasets()


def test_model_name_without_prefix():
    with pytest.raises(ValueError):
        ModelMeta(
            name="test_model",
            revision="test",
            release_date=None,
            languages=None,
            loader=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            reference=None,
            similarity_fn_name=None,
            use_instructions=None,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
        )


def test_model_training_dataset_adapted():
    model_meta = mteb.get_model_meta("deepvk/USER-bge-m3")
    assert model_meta.adapted_from == "BAAI/bge-m3"
    # MIRACLRetrieval not in training_datasets of deepvk/USER-bge-m3, but in
    # training_datasets of BAAI/bge-m3
    assert "MIRACLRetrieval" in model_meta.get_training_datasets()


@pytest.mark.parametrize(
    ("model_name", "expected_memory"),
    [
        ("intfloat/e5-mistral-7b-instruct", 13563),  # multiple safetensors
        ("NovaSearch/jasper_en_vision_language_v1", 3802),  # bf16
        ("intfloat/multilingual-e5-small", 449),  # safetensors
        ("BAAI/bge-m3", 2167),  # pytorch_model.bin
    ],
)
def test_model_memory_usage(model_name: str, expected_memory: int | None):
    meta = mteb.get_model_meta(model_name)
    assert meta.memory_usage_mb is not None
    used_memory = round(meta.memory_usage_mb)
    assert used_memory == expected_memory


def test_model_memory_usage_api_model():
    meta = mteb.get_model_meta("openai/text-embedding-3-large")
    assert meta.memory_usage_mb is None


@pytest.mark.parametrize("model_meta", mteb.get_model_metas())
def test_check_model_name_and_revision(model_meta: ModelMeta):
    assert model_meta.name is not None
    assert model_meta.revision is not None


@pytest.mark.parametrize("model_meta", mteb.get_model_metas())
def test_check_training_datasets_can_be_derived(model_meta: ModelMeta):
    # E.g. if a model if adapted_from is set to the model itself, this would cause infinite recursion. This ensures that this attribute can be called
    # without issues.
    # https://github.com/embeddings-benchmark/mteb/pull/3565
    assert model_meta.name != model_meta.adapted_from, (
        f"Model name and adapter model should be different. Got {model_meta.name} and {model_meta.adapted_from}"
    )
    model_meta.get_training_datasets()


@pytest.mark.parametrize("model_type", ["dense", "cross-encoder", "late-interaction"])
def test_get_model_metas_each_model_type(model_type):
    """Test filtering by each individual model type."""
    models = mteb.get_model_metas(model_types=[model_type])

    for model in models:
        assert model_type in model.model_type


def test_loader_kwargs_persisted_in_metadata():
    model = mteb.get_model(
        "baseline/random-encoder-baseline",
        not_existing_param=123,
    )

    assert hasattr(model, "mteb_model_meta")
    meta = model.mteb_model_meta

    assert "not_existing_param" in meta.loader_kwargs
    assert meta.loader_kwargs["not_existing_param"] == 123


def test_get_model_kwargs_does_not_mutate_registry_meta():
    model_name = "baseline/random-encoder-baseline"

    model = mteb.get_model(model_name, not_existing_param=123)
    assert model.mteb_model_meta.experiment_params == {"not_existing_param": 123}

    current_registry_meta = mteb.get_model_meta(model_name)
    assert current_registry_meta.experiment_params is None


def test_fill_missing_parameter():
    """Test that fill_missing parameter fetches missing metadata from HuggingFace Hub"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    meta_with_compute = mteb.get_model_meta(model_name, fill_missing=True)

    assert meta_with_compute.n_parameters is not None
    assert meta_with_compute.memory_usage_mb is not None


@pytest.mark.parametrize("model_meta", mteb.get_model_metas())
def test_n_embedding_parameters(model_meta: ModelMeta):
    """
    Test that tracks models with n_embedding_parameters=None.
    Historic models (in _HISTORIC_MODELS) are allowed to have None values.
    New models must have n_embedding_parameters defined, otherwise the test fails.
    """
    if model_meta.name in _MISSING_N_EMBEDDING_MODELS:
        assert model_meta.n_embedding_parameters is None
    else:
        assert model_meta.n_embedding_parameters is not None, (
            f"New model '{model_meta.name}' must have n_embedding_parameters defined. "
            f"If this is a historic model, add it to _HISTORIC_MODELS in test_model_meta.py"
        )


def test_model_to_python():
    meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")
    assert meta.to_python() == (
        """ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs={},
    name='sentence-transformers/all-MiniLM-L6-v2',
    revision='8b3219a92973c328a8e22fadcfa821b5dc75636a',
    release_date='2021-08-30',
    languages=['eng-Latn'],
    n_parameters=22713216,
    n_active_parameters_override=None,
    n_embedding_parameters=11720448,
    memory_usage_mb=87.0,
    max_tokens=256.0,
    embed_dim=384,
    license='apache-2.0',
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=['Sentence Transformers', 'PyTorch', 'ONNX', 'safetensors', 'Transformers'],
    reference='https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={'MSMARCO', 'MSMARCO-PL', 'MSMARCOHardNegatives', 'NQ', 'NQ-NL', 'NQ-PL', 'NQHardNegatives', 'NanoMSMARCORetrieval', 'NanoNQRetrieval', 'mMARCO-NL'},
    adapted_from=None,
    superseded_by=None,
    modalities=['text'],
    model_type=['dense'],
    citation=\'@inproceedings{reimers-2019-sentence-bert,\\n    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",\\n    author = "Reimers, Nils and Gurevych, Iryna",\\n    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",\\n    month = "11",\\n    year = "2019",\\n    publisher = "Association for Computational Linguistics",\\n    url = "http://arxiv.org/abs/1908.10084",\\n}\\n\',
    contacts=None,
)"""
    )


def test_model_meta_local_path():
    meta = ModelMeta.from_hub("/path/to/local/model")
    assert meta.name == "/path/to/local/model"
    assert meta.revision == "no_revision_available"


def test_load_cross_encoder_via_get_model_meta():
    """Test loading cross-encoder via get_model_meta() with automatic detection."""
    model_meta = mteb.get_model_meta("cross-encoder/ms-marco-TinyBERT-L-2-v2")

    assert model_meta.model_type == ["cross-encoder"]
    assert model_meta.is_cross_encoder
    assert model_meta.loader.__name__ == "CrossEncoderWrapper"


def test_load_sentence_transformer_via_get_model_meta():
    """Test loading sentence transformer via get_model_meta()."""
    model_meta = mteb.get_model_meta("sentence-transformers/all-MiniLM-L6-v2")

    assert model_meta.model_type == ["dense"]
    assert not model_meta.is_cross_encoder
    assert model_meta.loader.__name__ == "sentence_transformers_loader"
