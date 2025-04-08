import torch
from typing import Optional, Any
from pytorch_lightning import LightningDataModule

from ebr.datasets import get_retrieval_dataset
from ebr.utils.data import EmptyDataset, JSONLDataset


class EmbeddingDataCollator:

    def __call__(self, examples):
        assert len(examples) > 0
        batch = {
            key: [example[key] for example in examples]
            for key in examples[0].keys()
        }
        batch["embd"] = torch.tensor(batch["embd"])
        return batch


class RetrieveDataCollator:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._early_truncate = True

    def __call__(self, examples):
        assert len(examples) > 0
        batch = {}
        batch["id"] = [ex["id"] for ex in examples]
        batch["text"] = [ex["text"] for ex in examples]

        if self.tokenizer:
            texts = [s.strip() for s in batch["text"]]

            if self._early_truncate:
                max_str_len = self.tokenizer.model_max_length * 6
                texts = [s[:max_str_len] for s in texts]
 
            batch["input"] = self.tokenizer(
                texts,
                padding=True, 
                truncation=True, 
                return_tensors="pt",
            )

        return batch


class RetrieveDataModule(LightningDataModule):

    def __init__(
        self, 
        data_path: str,
        dataset_name: str,
        batch_size: int = 32, 
        embd_batch_size: int = 1024, 
        num_workers: int = 4,
        dataset_kwargs: Optional[dict] = None,
        collator_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.embd_batch_size = embd_batch_size
        self.num_workers = num_workers
        self.dataset = get_retrieval_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            **dataset_kwargs,
        )
        self.query_collator = None
        self.corpus_collator = None

    def prepare_data(self):
        self.dataset.prepare_data()

    def queries_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.queries,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.query_collator,
        )

    def corpus_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset.corpus, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=self.corpus_collator,
        )

    def set_queries_embds(self, queries_embds=None, queries_embds_files=None):
        if queries_embds:
            self.queries_embds = queries_embds
            self.queries_embd_ds = EmptyDataset(queries_embds)
        else:
            self.queries_embd_ds = JSONLDataset(queries_embds_files)
        assert len(self.queries_embd_ds) == len(self.dataset.queries)

    def set_corpus_embds(self, corpus_embds=None, corpus_embds_files=None):
        if corpus_embds:
            self.corpus_embds = corpus_embds
            self.corpus_embd_ds = EmptyDataset(corpus_embds)
        else:
            self.corpus_embd_ds = JSONLDataset(corpus_embds_files)
        # TODO: check this assertion later, removed for chunk model
        # assert len(self.corpus_embd_ds) == len(self.dataset.corpus)

    def queries_embd_dataloader(self):
        return torch.utils.data.DataLoader(
            self.queries_embd_ds,
            batch_size=self.embd_batch_size,
            num_workers=self.num_workers,
            collate_fn=EmbeddingDataCollator(),
        )

    def corpus_embd_dataloader(self):
        return torch.utils.data.DataLoader(
            self.corpus_embd_ds,
            batch_size=self.embd_batch_size,
            num_workers=self.num_workers,
            collate_fn=EmbeddingDataCollator(),
        )