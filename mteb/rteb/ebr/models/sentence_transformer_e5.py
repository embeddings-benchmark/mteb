from ebr.models import SentenceTransformersEmbeddingModel


class E5EmbeddingModel(SentenceTransformersEmbeddingModel):
    @property
    def model_name_prefix(self) -> str:
        return "intfloat"



# e5_mistral_7b_instruct = ModelMeta(
#     loader=SentenceTransformersEmbeddingModel,
#     model_name="e5-mistral-7b-instruct",
#     embd_dtype="float32",
#     embd_dim=4096,
#     similarity="cosine",
#     reference="https://huggingface.co/intfloat/e5-mistral-7b-instruct"
# )
