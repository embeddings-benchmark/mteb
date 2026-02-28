import io

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

import mteb
from mteb.models import EncoderProtocol
from mteb.models.model_meta import ModelMeta
from mteb.tasks import Vidore3HrRetrieval


class DoclingWrapper(EncoderProtocol):
    def __init__(self, *args, encoder: EncoderProtocol, **kwargs) -> None:
        self.encoder = encoder

    def encode(self, inputs, **kwargs):
        # first we OCR, then we encode
        id2markdown = self.built_id2markdown(inputs)

        # fetch the relevant markdowns using the id during the encode step - we don't load the entire thing into memory as some
        # datasets might not allow that, but we assumed that we can keep the markdowns in memory.
        _collate_fn = inputs.collate_fn
        inputs.collate_fn = self.add_ocr_collate_fn(_collate_fn, id2markdown)
        return self.encoder.encode(inputs, **kwargs)

    def add_ocr_collate_fn(self, collate_fn, id2markdown):
        def _collate_fn(*args, **kwargs):
            batch = collate_fn(*args, **kwargs)
            if "text" in batch:
                return batch
            if "image" in batch:
                batch["text"] = [id2markdown[_id] for _id in batch["id"]]
            return batch

        return _collate_fn

    def built_id2markdown(self, inputs):
        converter = DocumentConverter()
        id2markdown = {}
        for batch in inputs:
            streams = []
            for i, image in enumerate(batch["image"]):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                buf.seek(0)
                streams.append(DocumentStream(name=f"{i}.png", stream=buf))

            results = converter.convert_all(streams)
            for img_id, result in zip(batch["id"], results):
                id2markdown[img_id] = result.document.export_to_markdown()

        return id2markdown

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
model_w_ocr = DoclingWrapper(encoder=model)
task = Vidore3HrRetrieval()

# hasn't been run because it is too slow
res = mteb.evaluate(model_w_ocr, task, cache=None)
res[0].get_score()
res[0].evaluation_time  # 165.57329511642456

task.load_data()


image = task.dataset["english"]["test"]["corpus"][0]["image"]
type(image)  # PIL.PngImagePlugin.PngImageFile
converter = DocumentConverter()  # docling

# how to convert to to markdown

buf = io.BytesIO()
image.save(buf, format="PNG")
buf.seek(0)

stream = DocumentStream(name="image.png", stream=buf)
result = converter.convert(stream)
markdown = result.document.export_to_markdown()
