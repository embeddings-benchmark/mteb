from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import Any
import torch
from tqdm import tqdm
    
class CLIPModelWrapper:
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def preprocess(
            self,
            texts: list[str],
            images: list[Image.Image],
    ):
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)

    def get_text_embeddings(self, texts: list[str], batch_size: int = 32):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_outputs = self.model.get_text_features(**inputs)
                all_text_embeddings.append(text_outputs.cpu())

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        return all_text_embeddings

    def get_image_embeddings(self, images: list[Image.Image], batch_size: int = 32):
        all_image_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = images[i:i+batch_size]
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_outputs = self.model.get_image_features(**inputs)
                all_image_embeddings.append(image_outputs.cpu())

        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        return all_image_embeddings
    
    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits*100).softmax(dim=-1)
        return probs