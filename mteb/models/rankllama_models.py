import logging
import torch
from typing import List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from functools import partial
from mteb.models.rerankers_custom import RerankerWrapper
from typing import Any, Callable, List, Tuple
from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta

# vLLM imports
from vllm import LLM, SamplingParams, ModelRegistry
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.adapters import as_classification_model
from vllm.config import PoolerConfig

from mteb.models.rerankers_custom import RerankerWrapper

# Register LLaMA classification model for vLLM
LlamaForSequenceClassification = as_classification_model(LlamaForCausalLM)
ModelRegistry.register_model("LlamaForSequenceClassification", LlamaForSequenceClassification)


logger = logging.getLogger(__name__)

class RankLLaMA(RerankerWrapper):
    def __init__(
        self,
        model_name_or_path: str = "orionweller/rankllama-7b-merged",
        batch_size: int = 32, # NOTE: this is ignored
        context_size: int = 4096,
        fp_options: str = "float16",
        num_gpus: int = 1,
        **kwargs,
    ):
        super().__init__(model_name_or_path, batch_size=batch_size, fp_options=fp_options, **kwargs)
        self.context_size = context_size
        self.num_gpus = num_gpus

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._vllm_engine = LLM(
            model=model_name_or_path,
            task="classify",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=context_size,
            dtype=fp_options,
            trust_remote_code=True,
            max_model_len=context_size,
            override_pooler_config=PoolerConfig(
                softmax=False,
                pooling_type="LAST",
                normalize=False
            )
        )

    def truncate_inputs(self, inputs: List[str]) -> List[str]:
        truncated_inputs = []
        for input in inputs:
            # tokenize the input
            tokens = self.tokenizer.encode(input)
            if len(tokens) > self.context_size:
                tokens = tokens[:self.context_size-2]
                input = self.tokenizer.decode(tokens)
            truncated_inputs.append(input)
        return truncated_inputs
        
    def predict(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[int], List[float]]:
        if isinstance(prompts[0][1], dict):
            docs = [f"{doc[1]['title'] if 'title' in doc[1] else ''} {doc[1]['text']}".strip() for doc in prompts]
        else:
            docs = [prompt[1] for prompt in prompts]

        inputs = [f'query: {tuple_input[0].strip()}<s> document: {doc}' for (tuple_input, doc) in zip(prompts, docs)]
        inputs = self.truncate_inputs(inputs) # vLLM can't truncate and LLaMA 2 models can't handle them
        # print the first example
        print(inputs[0])
        outputs = self._vllm_engine.classify(inputs)
        scores = [item.outputs.probs[0] for item in outputs]
        return scores
    

def _loader(wrapper: type[RerankerWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner()


rankllama_7b = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=RankLLaMA,
        model_name_or_path="orionweller/rankllama-7b-merged",
        fp_options="float16",
    ),
    name="orionweller/rankllama-7b-merged",
    languages=["eng_Latn"],
    open_weights=True,
    revision="3ad9d39d869f42e2d89db6b7c207eb5359550439",
    release_date="2023-10-12",
    citation="""@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}"""
)


rankllama_14b = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=RankLLaMA,
        model_name_or_path="orionweller/rankllama-14b-merged",
        fp_options="float16",
    ),
    name="orionweller/rankllama-14b-merged",
    languages=["eng_Latn"],
    open_weights=True,
    revision="2338313e194a98e57a0f0fff07a062e1857dabf0",
    release_date="2023-10-12",
    citation="""@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}"""
)


