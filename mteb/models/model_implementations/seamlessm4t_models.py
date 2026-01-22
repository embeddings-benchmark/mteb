import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2Model

from mteb import TaskMetadata
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import Array, BatchedInput, PromptType
from mteb.types._encoder_io import AudioInput


class SeamlessM4TWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 5.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.model = SeamlessM4Tv2Model.from_pretrained(model_name, revision=revision)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        self.speech_encoder = self.model.speech_encoder

        self.model = self.model.to(device)
        self.speech_encoder = self.speech_encoder.to(device)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        import torchaudio

        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            audio_arrays = []
            max_samples = int(self.max_audio_length_seconds * self.sampling_rate)

            for a in batch["audio"]:
                array = torch.tensor(a["array"], dtype=torch.float32)
                sr = a.get("sampling_rate", None)
                if sr is None:
                    warnings.warn(
                        f"No sampling_rate provided for an audio sample. "
                        f"Assuming {self.sampling_rate} Hz (model default)."
                    )
                    sr = self.sampling_rate

                if sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.sampling_rate
                    )
                    array = resampler(array)

                # Squeeze to 1D and truncate
                array = array.squeeze()[:max_samples]
                audio_arrays.append(array)

            # Process the entire batch at once
            features = self.processor(
                audios=audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
            )

            input_features = features.input_features.to(self.device)
            attention_mask = (
                features.attention_mask.to(self.device)
                if hasattr(features, "attention_mask")
                else None
            )

            with torch.no_grad():
                outputs = self.speech_encoder(
                    input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )

                last_hidden_state = outputs.last_hidden_state

                # Apply attention-masked pooling to exclude padding tokens
                if attention_mask is not None:
                    batch_size, hidden_seq_len, hidden_size = last_hidden_state.shape
                    device = last_hidden_state.device

                    # For SeamlessM4T, check if attention mask matches hidden state length
                    if attention_mask.shape[1] != hidden_seq_len:
                        # Calculate downsample ratio and proper hidden lengths
                        input_lengths = attention_mask.sum(dim=1)
                        downsample_ratio = attention_mask.shape[1] / hidden_seq_len
                        hidden_lengths = (
                            input_lengths.float() / downsample_ratio
                        ).long()
                        hidden_lengths = torch.clamp(
                            hidden_lengths, min=0, max=hidden_seq_len
                        )

                        # Create attention mask for hidden states
                        seq_range = torch.arange(
                            hidden_seq_len, device=device
                        ).unsqueeze(0)
                        hidden_attention_mask = (
                            seq_range < hidden_lengths.unsqueeze(1)
                        ).to(last_hidden_state.dtype)
                    else:
                        # Use the attention mask directly if dimensions match
                        hidden_attention_mask = attention_mask.to(
                            last_hidden_state.dtype
                        )

                    # Apply masked mean pooling
                    hidden_attention_mask = hidden_attention_mask.unsqueeze(-1)
                    masked_embeddings = last_hidden_state * hidden_attention_mask
                    valid_tokens = hidden_attention_mask.sum(dim=1)
                    embeddings = masked_embeddings.sum(dim=1) / valid_tokens.clamp(
                        min=1e-9
                    )
                else:
                    # Fallback to simple mean pooling if no attention mask
                    embeddings = last_hidden_state.mean(dim=1)

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

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
        if "audio" not in inputs.dataset.features:
            raise ValueError("ASTWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


seamless_m4t_v2_large = ModelMeta(
    loader=SeamlessM4TWrapper,
    name="facebook/seamless-m4t-v2-large",
    languages=[
        # multilingual: supported languages can be found in the reference
        "eng-Latn"
    ],
    open_weights=True,
    revision="5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
    release_date="2023-11-06",
    max_tokens=None,
    n_parameters=2_300_000_000,
    memory_usage_mb=8809,
    embed_dim=1024,
    license="mit",
    reference="https://huggingface.co/facebook/seamless-m4t-v2-large",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/seamless_communication",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
    citation="""
@misc{communication2023seamlessmultilingualexpressivestreaming,
      title={Seamless: Multilingual Expressive and Streaming Speech Translation},
      author={Seamless Communication and Loïc Barrault and Yu-An Chung and Mariano Coria Meglioli and David Dale and Ning Dong and Mark Duppenthaler and Paul-Ambroise Duquenne and Brian Ellis and Hady Elsahar and Justin Haaheim and John Hoffman and Min-Jae Hwang and Hirofumi Inaguma and Christopher Klaiber and Ilia Kulikov and Pengwei Li and Daniel Licht and Jean Maillard and Ruslan Mavlyutov and Alice Rakotoarison and Kaushik Ram Sadagopan and Abinesh Ramakrishnan and Tuan Tran and Guillaume Wenzek and Yilin Yang and Ethan Ye and Ivan Evtimov and Pierre Fernandez and Cynthia Gao and Prangthip Hansanti and Elahe Kalbassi and Amanda Kallet and Artyom Kozhevnikov and Gabriel Mejia Gonzalez and Robin San Roman and Christophe Touret and Corinne Wong and Carleigh Wood and Bokai Yu and Pierre Andrews and Can Balioglu and Peng-Jen Chen and Marta R. Costa-jussà and Maha Elbayad and Hongyu Gong and Francisco Guzmán and Kevin Heffernan and Somya Jain and Justine Kao and Ann Lee and Xutai Ma and Alex Mourachko and Benjamin Peloquin and Juan Pino and Sravya Popuri and Christophe Ropers and Safiyyah Saleem and Holger Schwenk and Anna Sun and Paden Tomasello and Changhan Wang and Jeff Wang and Skyler Wang and Mary Williamson},
      year={2023},
      eprint={2312.05187},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.05187},
}""",
)
