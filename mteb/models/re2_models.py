from functools import partial
import logging
import math
from typing import Any, Callable, List, Tuple

import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
from mteb.model_meta import ModelMeta
from mteb.models.rerankers_custom import RerankerWrapper

logger = logging.getLogger(__name__)


class ReSquaredReranker(RerankerWrapper):
    name: str = "ReSquared"

    def __init__(
        self,
        model_name_or_path: str = "lightonai/re2-large",
        batch_size: int = 32,
        context_size: int = 4096,
        max_output_tokens: int = 1024,
        fp_options: str = "float16",
        use_vllm: bool = False,
        num_gpus: int = 1,
        **kwargs,
    ):
        super().__init__(model_name_or_path, batch_size=batch_size, fp_options=fp_options, **kwargs)
        
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.use_vllm = use_vllm
        self.num_gpus = num_gpus

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache commonly used token IDs
        self.true_token = self.tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer(" false", add_special_tokens=False).input_ids[0]
        self.think_token = self.tokenizer("<think>", add_special_tokens=False).input_ids[0]
        self.think_end_token = self.tokenizer("</think>", add_special_tokens=False).input_ids[-1]

        # Initialize model
        if self.use_vllm:
            self.model = LLM(
                model=model_name_or_path,
                tensor_parallel_size=num_gpus,
                dtype=fp_options,
                trust_remote_code=True,
                max_model_len=context_size,
                gpu_memory_utilization=0.9,
            )
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=max_output_tokens,
                logprobs=20,
                stop=["</think> true", "</think> false"],
                skip_special_tokens=False
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if fp_options == "float16" else torch.float32,
                device_map=self.device
            )
            self.model.eval()
            self.generation_config = GenerationConfig(
                temperature=0,
                max_new_tokens=max_output_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def _fix_incomplete_responses(
        self, 
        original_prompts: List[str], 
        generated_texts: List[str]
    ) -> Tuple[List[str], List[int], List[float]]:
        # Clean and prepare texts
        cleaned_texts = []
        for text in generated_texts:
            text = text.rstrip()
            if not text.endswith(('.', '!', '?')):
                last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last_punct != -1:
                    text = text[:last_punct + 1]
            cleaned_texts.append(text)
        
        forced_prompts = [
            f"{original_prompt}{cleaned_text}\n</think>" 
            for original_prompt, cleaned_text in zip(original_prompts, cleaned_texts)
        ]
        
        if self.use_vllm:
            outputs = self.model.generate(forced_prompts, self.sampling_params)
            return self._process_vllm_outputs(outputs, forced_prompts)
        else:
            inputs = self.tokenizer(
                forced_prompts, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            return self._process_hf_outputs(outputs, forced_prompts)

    def _process_vllm_outputs(self, outputs, prompts):
        final_texts = []
        token_counts = []
        scores = []
        
        for i, output in enumerate(outputs):
            try:
                token_count = len(output.outputs[0].token_ids)
                final_logits = output.outputs[0].logprobs[-1]
                
                if self.true_token not in final_logits or self.false_token not in final_logits:
                    score = 0.5
                    logger.warning(f"Incomplete response for prompt {i}")
                else:
                    true_logit = final_logits[self.true_token].logprob
                    false_logit = final_logits[self.false_token].logprob
                    true_score = math.exp(true_logit)
                    false_score = math.exp(false_logit)
                    score = true_score / (true_score + false_score)
            except:
                score = 0.5
                token_count = 0
                logger.warning(f"Error processing output for prompt {i}")
            
            final_text = f"{prompts[i]} {'true' if score >= 0.5 else 'false'}"
            final_texts.append(final_text)
            token_counts.append(token_count)
            scores.append(score)
        
        return final_texts, token_counts, scores

    def _process_hf_outputs(self, outputs, prompts):
        final_texts = []
        token_counts = []
        scores = []
        
        for i, sequence in enumerate(outputs.sequences):
            token_count = len(sequence)
            final_logits = outputs.scores[-1][i]
            true_false_logits = final_logits[[self.true_token, self.false_token]]
            probs = torch.softmax(true_false_logits, dim=-1)
            score = probs[0].item()
            
            final_text = f"{prompts[i]} {'true' if score >= 0.5 else 'false'}"
            final_texts.append(final_text)
            token_counts.append(token_count)
            scores.append(score)
        
        return final_texts, token_counts, scores

    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() if q.strip() != i.strip() else q.strip() for i, q in zip(instructions, queries)]

        if isinstance(passages[0], dict):
            passages = [f"{v['title']} {v['text']}" if 'title' in v else v['text'] for v in passages]
            # truncate to 1600 words
            final_passages = []
            for passage in passages:
                words = passage.split()
                if len(words) > 1600:
                    final_passages.append(" ".join(words[:1600]))
                else:
                    final_passages.append(passage)
            passages = final_passages

        prompts = [
            f"Determine if the following passage is relevant to the query. "
            f"Answer only with 'true' or 'false'.\n"
            f"Query: {query}\n"
            f"Passage: {passage}\n"
            f"<think>"
            for query, passage in zip(queries, passages)
        ]

        if self.use_vllm:
            outputs = self.model.generate(prompts, self.sampling_params)
            _, _, scores = self._process_vllm_outputs(outputs, prompts)
        else:
            inputs = self.tokenizer(
                prompts, 
                padding=True, 
                return_tensors="pt",
                truncation=True,
                max_length=self.context_size
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            _, _, scores = self._process_hf_outputs(outputs, prompts)

        return scores


class ReSquaredRerankerClassifier(RerankerWrapper):
    name: str = "ReSquaredClassifier"

    def __init__(
        self,
        model_name_or_path: str = "lightonai/re2-large",
        batch_size: int = 32,
        context_size: int = 4096,
        max_output_tokens: int = 1024,
        fp_options: str = "float16",
        use_vllm: bool = False,
        num_gpus: int = 1,
        **kwargs,
    ):
        super().__init__(model_name_or_path, batch_size=batch_size, fp_options=fp_options, **kwargs)
        
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.use_vllm = use_vllm
        self.num_gpus = num_gpus

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache commonly used token IDs
        self.true_token = self.tokenizer(" true", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer(" false", add_special_tokens=False).input_ids[0]
        self.think_token = self.tokenizer("<think>", add_special_tokens=False).input_ids[0]
        self.think_end_token = self.tokenizer("</think>", add_special_tokens=False).input_ids[-1]

        self.classifier = AutoModelForSequenceClassification.from_pretrained("/home/oweller2/my_scratch/rank_llm/results/checkpoint-1000")
        self.classifier_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        self.classifier.to(self.device)

        # Initialize model
        if self.use_vllm:
            self.model = LLM(
                model=model_name_or_path,
                tensor_parallel_size=num_gpus,
                dtype=fp_options,
                trust_remote_code=True,
                max_model_len=context_size,
                gpu_memory_utilization=0.9,
            )
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=max_output_tokens,
                logprobs=20,
                stop=["</think>"],
                skip_special_tokens=False
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if fp_options == "float16" else torch.float32,
                device_map=self.device
            )
            self.model.eval()
            self.generation_config = GenerationConfig(
                temperature=0,
                max_new_tokens=max_output_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

    def prepare_classifier_inputs(self, query, passage, output):
        # get text from vllm output
        reasoning = output.outputs[0].text
        text = f"[CLS] Query: {query} [SEP] Passage: {passage} [SEP] Reasoning: {reasoning.replace('</think>', '').replace('<think>', '').strip()} [SEP]"   
        return self.classifier_tokenizer(text, return_tensors="pt").to(self.device)

    def return_prompt(self, query, doc_content, is_plain: bool) -> str:
        if is_plain:
            return "Determine if the following passage is relevant to the query. " \
                   "Answer only with 'true' or 'false'.\n" \
                   f"Query: {query}\n" \
                   f"Passage: {doc_content}\n" \
                   "<think>" # force the model to start with this
        else:
            return f"""<|im_start|>system                                                                                                     
You are a helpful assistant.<|im_end|>                                                                                 
<|im_start|>user                                                                                                       
Determine if the following passage is relevant to the query. Answer only with 'true' or 'false'.                       
Query: {query}                                                                                                 
Passage: {doc_content}                                                                                                 
<|im_end|>                                                                                                             
<|im_start|>assistant                                                                                                  
<think>"""
    
    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() if q.strip() != i.strip() else q.strip() for i, q in zip(instructions, queries)]

        if isinstance(passages[0], dict):
            passages = [f"{v['title']} {v['text']}" if 'title' in v else v['text'] for v in passages]
            # truncate to 1600 words
            final_passages = []
            for passage in passages:
                words = passage.split()
                if len(words) > 1600:
                    final_passages.append(" ".join(words[:1600]))
                else:
                    final_passages.append(passage)
            passages = final_passages

        prompts = [
            self.return_prompt(query, passage, is_plain=True)
            for query, passage in zip(queries, passages)
        ]

        if self.use_vllm:
            outputs = self.model.generate(prompts, self.sampling_params)
        else:
            inputs = self.tokenizer(
                prompts, 
                padding=True, 
                return_tensors="pt",
                truncation=True,
                max_length=self.context_size
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

        # score with classifier
        scores = []
        for query, passage, reasoning in tqdm.tqdm(zip(queries, passages, outputs)):
            inputs = self.prepare_classifier_inputs(query, passage, reasoning)
            outputs = self.classifier(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            scores.append(outputs.logits[0][1].item())

        return scores


def _loader(wrapper: type[RerankerWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner()


# Model metadata for different Re² variants
re2_base = ModelMeta(
    loader=partial(
        _loader,
        wrapper=ReSquaredReranker,
        model_name_or_path="/home/oweller2/my_scratch/LLaMA-Factory/models/mistral-24b",
        fp_options="float16",
        use_vllm=True,
    ),
    name="resquared_mistral_24b",
    languages=["eng_Latn"],  # Add other supported languages if known
    open_weights=True,
    revision="main",  # Update with correct revision hash when available
    release_date="2024-01-15",  # Update with correct release date
)

re2_classifier = ModelMeta(
    loader=partial(
        _loader,
        wrapper=ReSquaredRerankerClassifier,
        model_name_or_path="/home/oweller2/my_scratch/LLaMA-Factory/models/mistral-24b",
        fp_options="float16",
        use_vllm=True,
    ),
    name="resquared_mistral_24b_classifier",
    languages=["eng_Latn"],  # Add other supported languages if known
    open_weights=True,
    revision="main",  # Update with correct revision hash when available
    release_date="2024-01-15",  # Update with correct release date
)
