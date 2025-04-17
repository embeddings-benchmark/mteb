---
# For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---
<!-- adapted from https://github.com/huggingface/huggingface_hub/blob/v0.30.2/src/huggingface_hub/templates/datasetcard_template.md -->

# Dataset Card for {{ pretty_name | default("Dataset Name", true) }}

{{ dataset_summary | default("", true) }}

<-- Datasets want link to arxiv in readme to autolink dataset with paper -->
Reference: {{ dataset_reference | default("", true) }}

## Citation

```bibtex
{{ citation_bibtex }}
```
