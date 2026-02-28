from datasets import load_dataset

import mteb
from mteb.models import EncoderProtocol
from mteb.models.model_meta import ModelMeta
from mteb.tasks import Vidore3HrRetrieval

meta = Vidore3HrRetrieval.metadata


class VidoreOCRFetchWrapper(EncoderProtocol):
    def __init__(self, *args, encoder: EncoderProtocol, **kwargs) -> None:
        self.encoder = encoder
        self.task_to_ds = {
            "Vidore3HrRetrieval": "vidore/vidore_v3_hr"
        }  # just one dataset as an example - I don't believe we have OCR for the closed datasets oO
        self.id2markdown = None
        self.current_dataset = None

    def encode(self, inputs, *, task_metadata, **kwargs):
        """Encode document but do fetch the OCR first."""
        if task_metadata.name != self.current_dataset or self.id2markdown is None:
            # jsu
            self.id2markdown = self.load_id2markdown(task_metadata)

        _collate_fn = inputs.collate_fn
        inputs.collate_fn = self.add_ocr_collate_fn(_collate_fn)
        return self.encoder.encode(inputs, task_metadata=task_metadata, **kwargs)

    def add_ocr_collate_fn(self, collate_fn):
        def _collate_fn(*args, **kwargs):
            assert self.id2markdown is not None
            batch = collate_fn(*args, **kwargs)
            if "text" in batch:
                return batch
            if "image" in batch:
                batch["text"] = [self.id2markdown[_id] for _id in batch["id"]]
            return batch

        return _collate_fn

    def load_id2markdown(self, metadata):
        name = metadata.name

        ds = load_dataset(self.task_to_ds[name], "corpus", split="test")
        return {
            f"corpus-test-{example['corpus_id']}": example["markdown"] for example in ds
        }

    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Update model meta to reflect that images are now handled"""
        modalities = self.encoder.mteb_model_meta.modalities
        if "image" not in modalities:
            modalities += ["image"]
        return self.encoder.mteb_model_meta.model_copy(
            update={"modalities": modalities}
        )

    def similarity(self, *args, **kwargs):
        return self.encoder.similarity(*args, **kwargs)

    def similarity_pairwise(self, *args, **kwargs):
        return self.encoder.similarity_pairwise(*args, **kwargs)


# choose an text only models
model = mteb.get_model("sentence-transformers/static-similarity-mrl-multilingual-v1")
model_w_ocr = VidoreOCRFetchWrapper(encoder=model)
task = Vidore3HrRetrieval()
res = mteb.evaluate(model_w_ocr, task, cache=None)
res[0].get_score()  # np.float64(0.23597)
res[0].evaluation_time  # 165.57329511642456
