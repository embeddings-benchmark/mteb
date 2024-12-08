from __future__ import annotations

import json
import warnings
from pathlib import Path

import iso639
from huggingface_hub import HfApi, ModelCard, hf_hub_download
from tqdm import tqdm

from mteb.model_meta import ModelMeta

to_keep = [
    "Haon-Chen/speed-embedding-7b-instruct",
    "Gameselo/STS-multilingual-mpnet-base-v2",
    "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
    "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
    "Hum-Works/lodestone-base-4096-v1",
    "Jaume/gemma-2b-embeddings",
    "BeastyZ/e5-R-mistral-7b",
    "Lajavaness/bilingual-embedding-base",
    "Lajavaness/bilingual-embedding-large",
    "Lajavaness/bilingual-embedding-small",
    "Mihaiii/Bulbasaur",
    "Mihaiii/Ivysaur",
    "Mihaiii/Squirtle",
    "Mihaiii/Venusaur",
    "Mihaiii/Wartortle",
    "Mihaiii/gte-micro",
    "Mihaiii/gte-micro-v4",
    "OrdalieTech/Solon-embeddings-large-0.1",
    "Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka",
    "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet",
    "Omartificial-Intelligence-Space/Arabic-all-nli-triplet-Matryoshka",
    "Omartificial-Intelligence-Space/Arabic-labse-Matryoshka",
    "Omartificial-Intelligence-Space/Arabic-mpnet-base-all-nli-triplet",
    "Omartificial-Intelligence-Space/Marbert-all-nli-triplet-Matryoshka",
    "consciousAI/cai-lunaris-text-embeddings",
    "consciousAI/cai-stellaris-text-embeddings",
    "manu/bge-m3-custom-fr",
    "manu/sentence_croissant_alpha_v0.2",
    "manu/sentence_croissant_alpha_v0.3",
    "manu/sentence_croissant_alpha_v0.4",
    "thenlper/gte-base",
    "thenlper/gte-large",
    "thenlper/gte-small",
    "OrlikB/KartonBERT-USE-base-v1",
    "OrlikB/st-polish-kartonberta-base-alpha-v1",
    "sdadas/mmlw-e5-base",  # some models are monolingual adaptions of a another models (I would include them for now)
    "dwzhu/e5-base-4k",  # e.g. this is a long doc adaption of e5
    "sdadas/mmlw-e5-large",
    "sdadas/mmlw-e5-small",
    "sdadas/mmlw-roberta-base",
    "sdadas/mmlw-roberta-large",
    "izhx/udever-bloom-1b1",
    "izhx/udever-bloom-3b",
    "izhx/udever-bloom-560m",
    "izhx/udever-bloom-7b1",
    "avsolatorio/GIST-Embedding-v0",
    "avsolatorio/GIST-all-MiniLM-L6-v2",
    "avsolatorio/GIST-large-Embedding-v0",
    "avsolatorio/GIST-small-Embedding-v0",
    "bigscience/sgpt-bloom-7b1-msmarco",
    "aari1995/German_Semantic_STS_V2",
    "abhinand/MedEmbed-small-v0.1",
    "avsolatorio/NoInstruct-small-Embedding-v0",
    "brahmairesearch/slx-v0.1",
    "deepfile/embedder-100p",
    "deepvk/USER-bge-m3",
    "infgrad/stella-base-en-v2",
    "malenia1/ternary-weight-embedding",
    "omarelshehy/arabic-english-sts-matryoshka",
    "openbmb/MiniCPM-Embedding",
    "shibing624/text2vec-base-multilingual",
    "silma-ai/silma-embeddding-matryoshka-v0.1",
    "zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
]

lang_to_script = {
    "bam": "Latn",
    "zul": "Latn",
    "tsn": "Latn",
    "rus": "Cyrl",
    "mar": "Deva",
    "ori": "Orya",
    "swa": "Latn",
    "vie": "Latn",
    "nld": "Latn",
    "kan": "Knda",
    "yor": "Latn",
    "urd": "Arab",
    "guj": "Gujr",
    "eng": "Latn",
    "tso": "Latn",
    "zho": "Hans",  # Can also be "Hant" depending on region
    "deu": "Latn",
    "sna": "Latn",
    "nso": "Latn",
    "pol": "Latn",
    "sot": "Latn",
    "mal": "Mlym",
    "cat": "Latn",
    "lug": "Latn",
    "spa": "Latn",
    "wol": "Latn",
    "tum": "Latn",
    "xho": "Latn",
    "fra": "Latn",
    "tam": "Taml",
    "pan": "Guru",
    "twi": "Latn",
    "tel": "Telu",
    "ibo": "Latn",
    "kik": "Latn",
    "run": "Latn",
    "hin": "Deva",
    "ben": "Beng",
    "fon": "Latn",
    "ita": "Latn",
    "nya": "Latn",
    "aka": "Latn",
    "por": "Latn",
    "asm": "Beng",
    "eus": "Latn",
    "lin": "Latn",
    "nep": "Deva",
    "kin": "Latn",
    "ind": "Latn",
    "ara": "Arab",
}


def convert_code(code: str) -> str | None:
    """Converts between two-letter and three-letter language codes"""
    try:
        lang_code = iso639.Language.match(code).part3
        script = lang_to_script[lang_code]
        return f"{lang_code}_{script}"
    except Exception as e:
        print(f"Couldn't convert {code}, reason: {e}")
        return None


api = HfApi()


def get_embedding_dimensions(model_name: str) -> int | None:
    try:
        file_path = hf_hub_download(
            repo_id=model_name, filename="1_Pooling/config.json"
        )
        with open(file_path) as in_file:
            pooling_config = json.loads(in_file.read())
            return pooling_config.get("word_embedding_dimension", None)
    except Exception as e:
        print(f"Couldn't get embedding size for {model_name}, reason: {e}")
        return None


def get_max_token(model_name: str) -> int | None:
    try:
        file_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(file_path) as in_file:
            config = json.loads(in_file.read())
            return config.get("max_position_embeddings", None)
    except Exception as e:
        print(f"Couldn't get embedding size for {model_name}, reason: {e}")
        return None


def get_base_model(model_name: str) -> str | None:
    try:
        file_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(file_path) as in_file:
            config = json.loads(in_file.read())
            base_model = config.get("_name_or_path", None)
            if base_model != model_name:
                return base_model
            else:
                return None
    except Exception as e:
        print(f"Couldn't get base model for {model_name}, reason: {e}")
        return None


def model_meta_from_hf_hub(model_name: str) -> ModelMeta:
    try:
        card = ModelCard.load(model_name)
        card_data = card.data.to_dict()
        frameworks = ["PyTorch"]
        if card_data.get("library_name", None) == "sentence-transformers":
            frameworks.append("Sentence Transformers")
        languages = card_data.get("language", None)
        if isinstance(languages, str):
            languages = [languages]
        if languages is not None:
            languages = [convert_code(l) for l in languages]
            languages = [l for l in languages if l is not None]
        repo_info = api.repo_info(model_name)
        revision = repo_info.sha
        release_date = repo_info.created_at.strftime("%Y-%m-%d")
        try:
            n_parameters = repo_info.safetensors.total
        except Exception as e:
            print(f"Couldn't get model size for {model_name}, reason: {e}")
            n_parameters = None
        n_dimensions = get_embedding_dimensions(model_name)
        datasets = card_data.get("datasets", None)
        if isinstance(datasets, str):
            datasets = [datasets]
        if datasets is not None:
            training_datasets = {ds: ["train"] for ds in datasets}
        else:
            training_datasets = None
        return ModelMeta(
            name=model_name,
            revision=revision,
            release_date=release_date,
            languages=languages,
            license=card_data.get("license", None),
            framework=frameworks,
            n_parameters=n_parameters,
            public_training_data=bool(datasets),
            adapted_from=get_base_model(model_name),
            training_datasets=training_datasets,
            open_weights=True,
            superseded_by=None,
            max_tokens=get_max_token(model_name),
            embed_dim=n_dimensions,
            similarity_fn_name="cosine",
            reference=f"https://huggingface.co/{model_name}",
        )
    except Exception as e:
        warnings.warn(f"Failed to extract metadata from model: {e}.")
        return ModelMeta(
            name=model_name,
            revision=None,
            languages=None,
            release_date=None,
        )


def code_from_meta(meta: ModelMeta) -> str:
    template = "{variable_name} ={meta}\n"
    variable_name = meta.name.replace("/", "__").replace("-", "_").replace(".", "_")
    return template.format(variable_name=variable_name, meta=meta.__repr__())


def main():
    out_path = Path("mteb/models/misc_models.py")
    with open(out_path, "w") as out_file:
        out_file.write("from mteb.model_meta import ModelMeta\n\n")
        for model in tqdm(to_keep, desc="Generating metadata for all models."):
            meta = model_meta_from_hf_hub(model)
            out_file.write(code_from_meta(meta))


if __name__ == "__main__":
    main()
