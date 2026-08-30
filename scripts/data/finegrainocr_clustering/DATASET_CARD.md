---
license: cc0-1.0
task_categories:
- feature-extraction
language:
- en
- fr
tags:
- mteb
- image-text
- clustering
pretty_name: FineGrainOCR Image-Text Clustering
---

# FineGrainOCR image-text clustering subset

This is a deterministic, evaluation-only subset of
[FineGrainOCR](https://github.com/Tubbias/finegrainocr) for cross-modal
clustering in MTEB. Each test row contains a grocery-product image, OCR derived
from that image, and a product-class label registered by the checkout barcode
scanner.

## Benchmark input and output

The embedding model receives both `image` and `text` for each row and produces
one vector. The clustering evaluator fits MiniBatch K-means to all vectors using
the known number of classes (256). It does not give product labels to the model
or clustering algorithm. The predicted cluster assignments are compared with
`label` using V-measure.

The `sample_id` column is provenance metadata and is not model input.

## Construction

The source validation split is ranked independently within each class by
SHA-256 of `42 + class ID + sample stem`. Up to 20 non-empty OCR pairs are kept
per class. Images are resized so their longest edge is at most 512 pixels and
encoded as optimized JPEG at quality 90. Source members are decompressed and
verified against the official ZIP's size and CRC-32 metadata.

OCR digit sequences containing 8–14 digits, including sequences separated by
spaces or hyphens, are replaced by `[BARCODE]`. This prevents printed GTINs from
revealing the class identifier while retaining product names and other package
text.

The dataset is built by
`scripts/data/finegrainocr_clustering/create_data.py` in the MTEB repository.

## Provenance and license

- Source repository: <https://github.com/Tubbias/finegrainocr>
- Source commit: `9ce19719123fd33a994b103b6e91c37a640ce92b`
- Paper: <https://doi.org/10.1007/s00138-024-01549-9>
- Source license: CC0-1.0

## Build summary

```json
{{SUMMARY_JSON}}
```

## Citation

```bibtex
@article{hansen2024finegrainocr,
  title = {FineGrainOCR: A dataset for text recognition in the wild},
  journal = {Machine Vision and Applications},
  year = {2024},
  doi = {10.1007/s00138-024-01549-9}
}
```
