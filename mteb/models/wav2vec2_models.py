from mteb.models.wrapper import Wrapper
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
import tqdm 

class Wav2Vec2Wrapper(Wrapper):
    def init(self, 
             model_name: str = "facebook/wav2vec2_base_960h", 
             device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)

    def get_text_embeddings(self, sentences: list[str], batch_size: int):
        text_embeddings = []

        with torch.no_grad():
             for i in tqdm(range(0, len(sentences), batch_size)):
                batch_texts = sentences[i : i + batch_size]
                inputs = self.processor(
                    text=batch_texts, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.model.get_text_features(**inputs)
                text_embeddings.append(text_outputs.cpu())

        return torch.cat(text_embeddings, dim=0)

    def get_audio_embeddings(self, dataset, batch_size: int):
        audio_embeddings = [] 

    def calculate_probs(self, text_embeddings, audio_embeddings):
        pass 

wav2vec2_base_960h = ModelMeta(
    name="facebook/wav2vec2_base_960h",
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
    training_datasets={
    },
    memory_usage_mb=390,
)