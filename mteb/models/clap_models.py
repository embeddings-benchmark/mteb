from __future__ import annotations

from functools import partial
from typing import Any, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor

from mteb.model_meta import ModelMeta
from torch.utils.data import DataLoader
import math
import os

def custom_collate_fn(batch):
    return batch


class ClapWrapper:
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)

    def get_text_embeddings(self, sentences: List[str], batch_size: int):
        text_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), batch_size)):
                batch_texts = sentences[i : i + batch_size]

                input_ids = self.tokenizer.batch_encode_plus(batch_texts, return_tensors='pt', padding='max_length', truncation='longest_first', max_length=512)
                # print(input_ids.data.keys())
                text_outputs = self.model.get_text_features(input_ids=input_ids.data['input_ids'], attention_mask=input_ids.data['attention_mask'])
                text_embeddings.append(text_outputs.cpu())

        return torch.cat(text_embeddings, dim=0)

    def get_audio_embeddings(self, dataset, sampling_rate: int, batch_size: int):
        audio_embeddings = []
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 0),
        )
        with torch.no_grad():
            for batch_audios in tqdm(dataloader):
                raw_audios = list(a['array'] for a in batch_audios)
                batch_encoding = self.feature_extractor(raw_audios, sampling_rate=sampling_rate)

                batch_embeddings = self.model.get_audio_features(torch.Tensor(np.array(batch_encoding['input_features'])))
                audio_embeddings.append(batch_embeddings.cpu())

        return torch.cat(audio_embeddings, dim=0)

    def calculate_probs(self, text_embeddings, audio_embeddings):
        print(text_embeddings.shape, audio_embeddings.shape)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        audio_embeddings = audio_embeddings / audio_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(audio_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs


clap_hstat_unfused = ModelMeta(
    loader=partial(
        ClapWrapper,
        model_name="laion/clap-htsat-unfused",
    ),
    name="laion/clap-htsat-unfused",
    languages=[],
    open_weights=True,
    revision="183bb99aa7af74355fb58d16edf8c13ae7c5433e",
    release_date="2022-01-23",
    n_parameters=102 * 1e6,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/facebook/wav2vec2-base-960h",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,
    memory_usage_mb=390,
    training_datasets={},
)
