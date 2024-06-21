from __future__ import annotations

import logging
import platform
import time
from pathlib import Path

import GPUtil
import numpy as np
import psutil

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.MTEBResults import ScoresDict

from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


WORDS_IN_UGLY_DUCKLING = 3633


class AbsTaskSpeedTask(AbsTask):
    """Abstract class for speed tasks (e.g. CPU, GPU)."""

    num_loops = 7
    device = "cpu"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        """Reads the text 'The Ugly Duckling' as the `test` split with a `text` column."""
        if self.data_loaded:
            return
        file_path = Path(__file__).parent / "the_ugly_duckling.txt"
        with file_path.open("r") as f:
            text = f.read()
        self.dataset = {"test": {"text": text.split("\n\n")}}
        self.data_loaded = True

    def _get_time_taken(self, model: Encoder, data_split) -> float:
        start = time.time()
        model.encode(data_split["text"], device=self.device)
        time_taken = time.time() - start
        return time_taken

    def get_system_info(self) -> dict[str, str]:
        """Returns a dictionary with system information."""
        info = {}
        info["platform"] = platform.system()
        info["platform_release"] = platform.release()
        info["platform_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["processor"] = platform.processor()
        info["ram"] = (
            str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        )  ## Convert from Bytes
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["total_cores"] = psutil.cpu_count(logical=True)

        ## NOTE: Currently works on nvidia GPUs only.
        if self.device != "cpu":
            gpus = GPUtil.getGPUs() or []
            info["num_gpus"] = len(gpus)
            list_gpus = []
            for gpu in gpus:
                list_gpus.append(
                    {
                        "gpu_name": gpu.name,
                        "gpu_total_memory": f"{gpu.memoryTotal/1024.0} GB",
                    }
                )
            info["gpu_info"] = list_gpus
        return info

    def _evaluate_subset(
        self, model: EncoderWithQueryCorpusEncode | Encoder, data_split, **kwargs
    ) -> ScoresDict:
        model.encode(["encode this"], device=self.device)  # ensure model is loaded

        timings = []
        for _ in range(self.num_loops):
            time_taken = self._get_time_taken(model, data_split)
            timings.append(time_taken)

        time_mean = np.mean(timings)
        time_std = np.std(timings)
        scores = {
            "time_mean": time_mean,
            "time_std": time_std,
            "timings": timings,
            "avg_words_per_sec": WORDS_IN_UGLY_DUCKLING / time_mean,
            **self.get_system_info(),
        }
        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]
