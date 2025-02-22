from functools import partial
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType, AudioEncoder
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from mteb.model_meta import ModelMeta
from datasets import Audio

class Wav2vec2Wrapper(AudioEncoder):
    def __init__(
        self, 
        device: str | None = None,
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self.model_name = kwargs.get('model_name', 'facebook/wav2vec2-base')
        self.model_revision = kwargs.get('model_revision', None)
        
        self.model = Wav2Vec2Model.from_pretrained(
            self.model_name, 
            revision=self.model_revision
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, 
            revision=self.model_revision
        )
        self.embed_dim = self.model.config.hidden_size
        
        if device:
            self.model = self.model.to(device)
        print("Wav2vec initialized")
        
    def get_audio_embeddings(
        self, 
        audio_files: list[Audio] | Audio, 
        **kwargs
    ) -> np.ndarray:
        
        batch_size = kwargs.get('batch_size', 32)
        
        if not isinstance(audio_files, list):
            audio_files = [audio_files]
            
        all_embeddings = []
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            
            audio_data = [file['array'] for file in batch]
            sampling_rates = [file['sampling_rate'] for file in batch]
            
            # Preprocess batch
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=sampling_rates,
                padding=True,
                return_tensors="pt"
            )
            
            if hasattr(self, 'device') and self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)

    def encode(
        self,
        audio_files: list[Audio],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        return self.get_audio_embeddings(audio_files, **kwargs)



wav2vec2_base = ModelMeta(
    loader=partial(Wav2vec2Wrapper, model_name="facebook/wav2vec2-base"),
    name="facebook/wav2vec2-base",
    languages=["en"],
    open_weights=True,
    revision="main",         
    release_date="2020-10-26",
    max_tokens=float("inf"),
    n_parameters=95_000_000,
    memory_usage_mb=362,
    embed_dim=768,
    license="Apache-2.0",
    reference="https://huggingface.co/facebook/wav2vec2-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,    
    modalities=["audio"]
)
