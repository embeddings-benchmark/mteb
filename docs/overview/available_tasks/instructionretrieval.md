
# InstructionRetrieval

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** 8 

#### IFIRAila

Benchmark aila subset in aila within instruction following abilities. The instructions simulate lawyers' or legal assistants' nuanced queries to retrieve relevant legal documents. 

**Dataset:** [`if-ir/aila`](https://huggingface.co/datasets/if-ir/aila) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Legal, Written | human-annotated | found |



#### IFIRCds

Benchmark IFIR cds subset within instruction following abilities. The instructions simulate a doctor's nuanced queries to retrieve suitable clinical trails, treatment and diagnosis information. 

**Dataset:** [`if-ir/cds`](https://huggingface.co/datasets/if-ir/cds) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Medical, Written | human-annotated | found |



#### IFIRFiQA

Benchmark IFIR fiqa subset within instruction following abilities. The instructions simulate people's daily life queries to retrieve suitable financial suggestions. 

**Dataset:** [`if-ir/fiqa`](https://huggingface.co/datasets/if-ir/fiqa) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Financial, Written | human-annotated | created |



#### IFIRFire

Benchmark IFIR fire subset within instruction following abilities. The instructions simulate lawyers' or legal assistants' nuanced queries to retrieve relevant legal documents. 

**Dataset:** [`if-ir/fire`](https://huggingface.co/datasets/if-ir/fire) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Legal, Written | human-annotated | found |



#### IFIRNFCorpus

Benchmark IFIR nfcorpus subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature in the medical and biological domains. 

**Dataset:** [`if-ir/nfcorpus`](https://huggingface.co/datasets/if-ir/nfcorpus) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Academic, Medical, Written | human-annotated | found |



#### IFIRPm

Benchmark IFIR pm subset within instruction following abilities. The instructions simulate a doctor's nuanced queries to retrieve suitable clinical trails, treatment and diagnosis information. 

**Dataset:** [`if-ir/pm`](https://huggingface.co/datasets/if-ir/pm) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Medical, Written | human-annotated | found |



#### IFIRScifact

Benchmark IFIR scifact_open subset within instruction following abilities. The instructions in this dataset simulate nuanced queries from students or researchers to retrieve relevant science literature. 

**Dataset:** [`if-ir/scifact_open`](https://huggingface.co/datasets/if-ir/scifact_open) • **License:** mit • [Learn more →](https://arxiv.org/abs/2503.04644)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | ndcg_at_20 | eng | Academic, Written | human-annotated | found |



#### InstructIR

A benchmark specifically designed to evaluate the instruction following ability in information retrieval models. Our approach focuses on user-aligned instructions tailored to each query instance, reflecting the diverse characteristics inherent in real-world search scenarios. **NOTE**: scores on this may differ unless you include instruction first, then "[SEP]" and then the query via redefining `combine_query_and_instruction` in your model.

**Dataset:** [`mteb/InstructIR-mteb`](https://huggingface.co/datasets/mteb/InstructIR-mteb) • **License:** mit • [Learn more →](https://github.com/kaistAI/InstructIR/tree/main)

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| text to text (t2t) | robustness_at_10 | eng | Web | human-annotated | created |
