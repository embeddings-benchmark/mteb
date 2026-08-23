# Mock Run Results — judicialmind/greenleaf-law-embed-tiny

Model loads and encodes correctly through the custom `GreenLeafEmbedWrapper`.

```
Name: judicialmind/greenleaf-law-embed-tiny
Revision: 7939f1bf2945e06f75191daa80feee191af38d13
Parameters: 595,776,512
Embed dim: 1024
Max tokens: 32768
License: apache-2.0
Open weights: True
Framework: Sentence Transformers, PyTorch, safetensors
```

Verified:
- ModelMeta loads correctly
- `GreenLeafEmbedWrapper` wraps SentenceTransformer with `trust_remote_code=True`
- `mteb.evaluate()` runs successfully on MTEB(Law, v1) tasks
- AILAStatutes: 58.79 (matches expected score)
- All 8 MTEB(Law, v1) tasks complete without OOM on a single GH200 (97GB)
- Embedding shape: (batch, 1024), float32
- Mean pooling, cosine similarity
- No instruction prefix required
