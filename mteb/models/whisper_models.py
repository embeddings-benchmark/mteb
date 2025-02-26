from functools import partial
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType, AudioEncoder
import numpy as np
import torch
from transformers import WhisperModel, WhisperProcessor
from mteb.model_meta import ModelMeta
from datasets import Audio

class WhisperWrapper(AudioEncoder):
    def __init__(self,
                 model_name: str,
                 revision: str = "main",
                 device: str | None = None,
                 **kwargs):
        super().__init__(device=device, **kwargs)
        self.model_name = model_name
        self.model_revision = revision

        self.model = WhisperModel.from_pretrained(self.model_name, revision=self.model_revision)
        self.feature_extractor = WhisperProcessor.from_pretrained(self.model_name, revision=self.model_revision)
        self.embed_dim = self.model.config.d_model

        if device:
            self.model = self.model.to(device)
        print("Whisper model initialized.")

    def get_audio_embeddings(self,
                             audio_files: list[Audio] | Audio,
                             batch_size: int = 32,
                             **kwargs) -> np.ndarray:
        if not isinstance(audio_files, list):
            audio_files = [audio_files]

        all_embeddings = []
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            audio_data = [file['array'] for file in batch]
            sampling_rates = [file['sampling_rate'] for file in batch]

            # converts raw waveform to log-Mel spectrograms
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=sampling_rates[0],
                return_tensors="pt",
                padding="max_length",  # force padding to a fixed raw sample length
                max_length=480000      # 30 seconds * 16000 Hz => 480000 samples -> 480000/160 = 3000 mel frames (whisper expects 3000 frames)
            )

            if hasattr(self, 'device') and self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                encoder_outputs = self.model.encoder(inputs.input_features)

            embeddings = encoder_outputs.last_hidden_state
            batch_embeddings = embeddings.mean(dim=1).cpu().numpy()
            print(batch_embeddings.shape)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def encode(self,
               audio_files: list[Audio],
               *,
               task_name: str,
               prompt_type: PromptType | None = None,
               **kwargs) -> np.ndarray:
        return self.get_audio_embeddings(audio_files, **kwargs)




whisper_tiny = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-tiny"),
    name="openai/whisper-tiny",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=39_000_000,      
    memory_usage_mb=144,         
    embed_dim=512,              
    license="MIT",
    reference="https://huggingface.co/openai/whisper-tiny",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_base = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-base"),
    name="openai/whisper-base",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=74_000_000,      
    memory_usage_mb=277,          
    embed_dim=512,  
    license="MIT",
    reference="https://huggingface.co/openai/whisper-base",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_small = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-small"),
    name="openai/whisper-small",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=244_000_000,    
    memory_usage_mb=922,        
    embed_dim=768,         
    license="MIT",
    reference="https://huggingface.co/openai/whisper-small",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_medium = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-medium"),
    name="openai/whisper-medium",
    languages=["eng", "multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=769_000_000,     
    memory_usage_mb=2914,     
    embed_dim=1024,      
    license="MIT",
    reference="https://huggingface.co/openai/whisper-medium",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

whisper_large_v3 = ModelMeta(
    loader=partial(WhisperWrapper, model_name="openai/whisper-large-v3"),
    name="openai/whisper-large-v3",
    languages=["multilingual"],
    open_weights=True,
    revision="main",
    release_date="2022-09-27",
    max_tokens=float("inf"),
    n_parameters=1550_000_000,  
    memory_usage_mb=5887,   
    embed_dim=1280,           
    license="MIT",
    reference="https://huggingface.co/openai/whisper-large-v3",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"]
)

# print(f"whisper_tiny: {whisper_tiny.calculate_memory_usage_mb()}")
# print(f"whisper_base: {whisper_base.calculate_memory_usage_mb()}")
# print(f"whisper_small: {whisper_small.calculate_memory_usage_mb()}")
# print(f"whisper_medium: {whisper_medium.calculate_memory_usage_mb()}")
# print(f"whisper_large_v3: {whisper_large.calculate_memory_usage_mb()}")
