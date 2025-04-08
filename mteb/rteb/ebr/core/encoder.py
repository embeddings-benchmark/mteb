import os
import json
import logging
from typing import List

from pytorch_lightning import LightningModule

from ebr.core.base import EmbeddingModel
from ebr.utils.data import JSONLDataset
from ebr.utils.distributed import gather_list

logger = logging.getLogger(__name__)


class Encoder(LightningModule):

    def __init__(
        self,
        model: EmbeddingModel,
        save_embds: bool = False,
        load_embds: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._model = model
        self._load_embds = load_embds
        self._save_embds = save_embds
        # Keep the embeddings in memory by default. Set it to False for large corpus.
        self.in_memory = True
        self.is_query = False
        self.save_file = None

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    @property
    def load_embds(self) -> bool:
        return self._load_embds

    @property
    def save_embds(self) -> bool:
        # If in_memory=False, we have to save the embeddings
        return self._save_embds or not self.in_memory

    @property
    def local_embd_file_name(self) -> str:
        assert self.save_file is not None
        num_shards = self.trainer.num_devices
        return f"{self.save_file}-{self.local_rank}-of-{num_shards}"

    def get_local_embd_files(self, num_shards=None) -> List[str]:
        # Return local (intermediate) file names, which are jsonl files
        assert self.save_file is not None
        if num_shards is None:
            num_shards = self.trainer.num_devices
        return [f"{self.save_file}-{i}-of-{num_shards}" for i in range(num_shards)]
    
    def get_embd_files(self, num_shards=None) -> List[str]:
        # Return the final file names, which are arrow files
        local_files = self.get_local_embd_files(num_shards=num_shards)
        return local_files
    
    def embd_files_exist(self, num_shards=None) -> bool:
        files = self.get_embd_files(num_shards=num_shards)
        return all(os.path.exists(file) for file in files)

    def on_predict_epoch_start(self):
        self.embds = None

        if self.in_memory:
            self.local_embds = []

        if self.load_embds:
            self.local_existing_ids = set()
            if os.path.exists(self.local_embd_file_name):
                logger.warning(f"Load embeddings from {self.local_embd_file_name}")
                ds = JSONLDataset(self.local_embd_file_name)
                for example in ds:
                    self.local_existing_ids.add(example["id"])
                    if self.in_memory:
                        self.local_embds.append(example)
            else:
                logger.warning(
                    f"load_embds is True but {self.local_embd_file_name} doesn't exist. Skipping the loading.")

        if self.save_embds:
            if self.load_embds:
                # append to the file
                self.local_embd_file = open(self.local_embd_file_name, "a")
            else:
                # rewrite the file
                self.local_embd_file = open(self.local_embd_file_name, "w")

    def predict_step(self, batch, batch_idx):
        indices = batch["id"]
        
        if self.load_embds and self.local_existing_ids:
            masks = [id in self.local_existing_ids for id in indices]
            num_existed = sum(masks)
            if num_existed == len(indices):
                return
            elif num_existed > 0:
                raise NotImplementedError("Partial loading within batch is not supported yet.")

        embds = self._model(batch)

        for idx, embd in zip(indices, embds):
            obj = {"id": idx, "embd": embd}
            if self.in_memory:
                self.local_embds.append(obj)
            if self.save_embds:
                self.local_embd_file.write(json.dumps(obj) + "\n")

    def on_predict_epoch_end(self):
        if self.save_embds:
            self.local_embd_file.close()
        if self.in_memory:
            self.embds = gather_list(self.local_embds, self.trainer.num_devices)
        self.trainer.strategy.barrier()
