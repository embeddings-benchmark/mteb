---
title: "Image models on video tasks"
icon: lucide/film
---

## Evaluating Image Models on Video Tasks

Image-text models such as CLIP or SigLIP are commonly reported on video benchmarks even though they have no notion of time: a fixed number of frames is sampled uniformly from each clip, every frame is encoded as an image, and the frame embeddings are mean-pooled into a single video embedding. This is the protocol used for CLIP-style baselines in CLIP4Clip, X-CLIP and ChinaOpen, among others.

By default MTEB refuses to run a model on a task whose modalities it does not declare, so a model with `modalities=["image", "text"]` raises on a `["text", "video"]` task. The [`VideoFramesWrapper`][mteb.models.video_wrappers.video_frames_wrapper.VideoFramesWrapper] makes the frame-pooling protocol explicit:

```python
import mteb
from mteb.models import VideoFramesWrapper

task = mteb.get_task("MSRVTTT2V")
model = mteb.get_model("openai/clip-vit-base-patch32")

video_model = VideoFramesWrapper(model, num_frames=8)
results = mteb.evaluate(video_model, tasks=[task])
```

`num_frames` defaults to 8 and controls how many frames are sampled uniformly from each video. Text inputs (for example the queries of a text-to-video task) are passed through to the wrapped model unchanged.

### Provenance

The wrapper records `video_num_frames` and `video_frame_pooling` in the model's `experiment_kwargs`, and adds `"video"` to its declared modalities. Results produced this way are therefore distinguishable from those of native video models, and the frame count is always part of the result even when the default is used.

### Limitations

- Only video-only inputs are pooled. Tasks whose rows combine video with text or audio in a single input (e.g. `vt2t`) are not supported by the wrapper and raise `NotImplementedError`.
- Decoding video requires the `video` extra (`pip install "mteb[video]"`).
