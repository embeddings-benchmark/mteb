<h1 align="center">Gene-MTEB Benchmark</h1>

Gene-MTEB is a specialized extension of the [MTEB](https://github.com/mteb/mteb) repository, tailored for metagenomic analysis using gene sequences derived from the [Human Microbiome Project (HMP)](https://www.ncbi.nlm.nih.gov/bioproject/28331), [Human Virus Reference Sequences](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/) and Human Virus infecting samples.

Please refer to our Huggingface page to access all the related datasets: [metagene-ai](https://huggingface.co/metagene-ai). 

## Quick Tour

We add in total seven classification tasks, one multi-label classification task, and four clustering tasks to the benchmark.

**Classification tasks**:
- [HumanVirusClassification.py](mteb/tasks/Classification/metagene/HumanVirusClassification.py): four classification tasks using human virus infecting samples.
- [HumanMicrobiomeProjectDemonstrationClassification.py](mteb/tasks/Classification/metagene/HumanMicrobiomeProjectDemonstrationClassification.py): three classification tasks using HMP demonstration sequences.

**Multi-label classification task**:
- [HumanMicrobiomeProjectDemonstrationMultiLabelClassification.py](mteb/tasks/MultiLabelClassification/metagene/HumanMicrobiomeProjectDemonstrationMultiLabelClassification.py): one multi-label classification task using HMP demonstration sequences.

**Clustering tasks**:
- [HumanVirusReferenceClustering.py](mteb/tasks/Clustering/metagene/HumanVirusReferenceClustering.py): four clustering tasks using human virus reference sequences.
- [HumanMicrobiomeProjectReferenceClustering.py](mteb/tasks/Clustering/metagene/HumanMicrobiomeProjectReferenceClustering.py): four clustering tasks using HMP reference sequences.


## Installation

```bash
pip install torch transformers numpy tqdm
git clone https://github.com/metagene-ai/gene-mteb.git
cd gene-mteb && pip install -e .
```

## Example Using METAGENE-1

```python
import mteb
from mteb.encoder_interface import PromptType
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.trainer_utils import set_seed
import torch


class LlamaWrapper:
    def __init__(self,
                 model_name,
                 seed,
                 max_length=512):

        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "auto")

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.model.eval()

    def encode(self,
               sentences,
               task_name: str | None = None,
               prompt_type: PromptType | None = None,
               **kwargs):

        set_seed(self.seed)
        batch_size = kwargs.get("batch_size", 32)

        embeddings = []

        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.model.device)

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings.cpu().to(torch.float32).numpy())

        return  np.array(embeddings)

    
if __name__ == "__main__":
    model = LlamaWrapper(
        model_name="metagene-ai/METAGENE-1", 
        seed=42)

    tasks = mteb.get_tasks(tasks=[
        "HumanVirusClassificationOne",
        "HumanVirusClassificationTwo",
        "HumanVirusClassificationThree",
        "HumanVirusClassificationFour",
        "HumanMicrobiomeProjectDemonstrationClassificationDisease",
        "HumanMicrobiomeProjectDemonstrationClassificationSex",
        "HumanMicrobiomeProjectDemonstrationClassificationSource",
        "HumanMicrobiomeProjectDemonstrationMultiLabelClassification",
        "HumanVirusReferenceClusteringP2P",
        "HumanVirusReferenceClusteringS2SAlign",
        "HumanVirusReferenceClusteringS2SSmall",
        "HumanVirusReferenceClusteringS2STiny",
        "HumanMicrobiomeProjectReferenceClusteringP2P",
        "HumanMicrobiomeProjectReferenceClusteringS2SAlign",
        "HumanMicrobiomeProjectReferenceClusteringS2SSmall",
        "HumanMicrobiomeProjectReferenceClusteringS2STiny",
    ])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model)
```
