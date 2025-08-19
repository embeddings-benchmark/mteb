from __future__ import annotations

import json
import queue
import logging
import numpy as np
from typing import Any
from functools import partial
from tqdm.autonotebook import tqdm
from contextlib import nullcontext
from collections.abc import Sequence

import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data._utils.worker import ManagerWatchdog

import mteb
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType

logger = logging.getLogger(__name__)

def _init_worker(device_id):
    import torch
    import signal
    
    torch.cuda.set_device(device_id)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    torch.cuda.init()
    torch.manual_seed(torch.initial_seed() + device_id)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch.initial_seed() + device_id)

def _encode_loop(
    model: TransformersTextEmbedder,
    input_queue,
    output_queue,
    device: torch.device,
    qsize: int = 4,
    amp_dtype=None
):
    device_id = device.index if device.index is not None else 0
    _init_worker(device_id)
    
    try:
        model = model.to(device)
        watchdog = ManagerWatchdog()
        keep_queue = queue.Queue(qsize + 1)

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype
            ) if amp_dtype is not None else nullcontext():
                while watchdog.is_alive():
                    try:
                        r = input_queue.get()
                        if r is None:
                            break

                        n, inputs = r
                        embeddings = model.embed(*inputs, device=device)
                        output_queue.put((n, embeddings))
                        
                        if keep_queue.full():
                            i = keep_queue.get()
                            del i
                        keep_queue.put(embeddings)
                        del r, n, inputs
                    except Exception as e:
                        logger.error(f"Error in worker process: {str(e)}")
                        break

        while not keep_queue.empty():
            i = keep_queue.get()
            del i
            
    except Exception as e:
        logger.error(f"Worker process failed: {str(e)}")
    finally:
        torch.cuda.empty_cache()
        del model, watchdog
    return


class TransformersTextEmbedder(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pooler_type: str = 'last',
        do_norm: bool = False,
        truncate_dim: int = 0,
        padding_left: bool = False,
        attn_type: str = 'causal',
        **kwargs,
    ):
        super().__init__()
        self.pooling_method = "mean"
        self.normalize_embeddings = True
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")

    def embed(
        self, 
        sentences: Sequence[str], 
        max_length: int,
        prompt: str | None = None,
        device: str | torch.device = 'cpu',
    ) -> torch.Tensor:
        if prompt:
            sentences = [prompt + t for t in sentences]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
        ).to(device)

        offsets = inputs.pop("offset_mapping")

        with torch.no_grad():            
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            if self.pooling_method == 'mean':
                instruction_char_lens = [len(prompt)] * len(sentences)
            
                instruction_lens_tensor = torch.tensor(
                    instruction_char_lens, 
                    device=offsets.device
                ).unsqueeze(1)

                end_offsets = offsets[:, :, 1]
                pooling_mask = (end_offsets > instruction_lens_tensor).to(inputs["attention_mask"].dtype)
                
                pooling_mask = pooling_mask * inputs["attention_mask"]
                embeddings = self.mean_pooling(last_hidden_state, pooling_mask)
            elif self.pooling_method == 'cls':
                embeddings = last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling_method}.")
            
            if self.normalize_embeddings:
                in_dtype = embeddings.dtype
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)

        return embeddings

    @staticmethod
    def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        embedding = s / d
        return embedding

def _encode_loop(
    model: TransformersTextEmbedder,
    input_queue,
    output_queue,
    device: torch.device,
    qsize: int = 4,
    amp_dtype=None
):
    model = model.to(device)
    watchdog = ManagerWatchdog()
    keep_queue = queue.Queue(qsize + 1)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype
        ) if amp_dtype is not None else nullcontext():
            while watchdog.is_alive():
                r = input_queue.get()
                
                if r is None:
                    break

                n, inputs = r
                embeddings = model.embed(*inputs, device=device)
                output_queue.put((n, embeddings))
                
                if keep_queue.full():
                    i = keep_queue.get()
                    del i
                
                keep_queue.put(embeddings)
                del r, n, inputs

    while not keep_queue.empty():
        i = keep_queue.get()
        del i
    del model, watchdog
    return


class LlmEmbedding(Wrapper):
    _model_class = TransformersTextEmbedder

    def __init__(
        self,
        model: str,
        use_instruction: bool = False,
        device: str = 'cuda',
        max_length: int = 1024,
        max_query_length: int | None = None,
        max_doc_length: int | None = None,
        precision: str = 'fp32',
        mp_qsize: int = 4,
        instruction_dict_path=None,
        instruction_template=None,
        instruction_dict: dict | None = None,
        **kwargs,  # For `TransformersTextEmbedder`
    ) -> None:
        model_name = model.split('/')
        
        if model_name[-1] == '':
            model_name = model_name[-2]
        else:
            model_name = model_name[-1]
        
        model_name = kwargs.pop('model_name', model_name)
        self.model = self._model_class(model, **kwargs)

        self.mteb_model_meta = ModelMeta(
            name=model_name, revision=kwargs.get('revision', None), 
            release_date=None, languages=None, n_parameters=None, memory_usage_mb=None, 
            max_tokens=None, embed_dim=None, license=None, open_weights=False, public_training_code=None, 
            public_training_data=None, framework=["Sentence Transformers"], similarity_fn_name="cosine", use_instructions=True, training_datasets=None
        )

        self.device = device
        self.use_instruction = use_instruction
        self.max_doc_length = max_doc_length or max_length
        self.max_query_length = max_query_length or max_length
        
        self.amp_dtype = None
        if precision == 'fp16':
            self.model.half()
        elif precision == 'bf16':
            self.model.bfloat16()
        elif precision.startswith('amp_'):
            self.amp_dtype = torch.float16 if precision.endswith('fp16') else torch.bfloat16
        
        self.mp_qsize = mp_qsize
        n_gpu = torch.cuda.device_count()

        self.world_size = n_gpu
        assert n_gpu > 0, 'woho, no no no!'
        logger.info(f"We have {n_gpu=}, good.")
        
        self._workers = list()
        self._input_queues = list()
        self._output_queues = list()
        
        self.instruction_dict = instruction_dict.copy() if instruction_dict else {}
        if not self.instruction_dict and instruction_dict_path is not None:
            with open(instruction_dict_path) as f:
                self.instruction_dict = json.load(f)
        if instruction_template is not None:
            self.instruction_template = instruction_template

    def get_instruction(self, task_name, prompt_type):
        sym_task = False
        
        if task_name in self.instruction_dict:
            instruction = self.instruction_dict[task_name]
            if isinstance(instruction, dict):
                instruction = instruction.get(prompt_type, "")
                sym_task = True
        else:
            instruction = super().get_instruction(task_name, prompt_type)
        task_type = mteb.get_tasks(tasks=[task_name])[0].metadata.type
        
        if 'Retrieval' in task_type and not sym_task and prompt_type != 'query':
            return "<s>"
        
        if 'Retrieval' in task_type and prompt_type == 'query' and instruction is None:
            instruction = "Retrieval relevant passage for the given query."
        
        return instruction
        
    def format_instruction(self, instruction, prompt_type):
        if instruction is not None and len(instruction.strip()) > 0:
            instruction = self.instruction_template.format(instruction)
            return instruction
        return ""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        instruction = None
        if self.use_instruction:
            instruction = self.get_instruction(task_name, prompt_type)
            if self.instruction_template:
                instruction = self.format_instruction(instruction, prompt_type)
            logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")

        num_texts = len(sentences)
        logger.info(f"Encoding {num_texts} sentences. samples: {[text[:50] for text in sentences[:3]]} ...")
        num_batches = num_texts // batch_size + int(num_texts % batch_size > 0)

        def _receive(oq, timeout=0.00125):
            try:
                n, embed = oq.get(timeout=timeout)
                result_dict[n] = embed.cpu()
                pbar.update(1)
                del embed
            except queue.Empty:
                pass

        max_length = self.max_query_length if prompt_type == PromptType.query else self.max_doc_length

        pbar = tqdm(
            total=num_batches, disable=not show_progress_bar, desc='encode',
            mininterval=1, miniters=10
        )
        result_dict = dict()
        if not self._workers:
            self.model.to(self.device)

        with nullcontext() if self._workers else torch.inference_mode():
            with nullcontext() if self._workers or self.amp_dtype is None else torch.autocast(
                device_type=self.device, dtype=self.amp_dtype
            ):
                for n, i in enumerate(range(0, num_texts, batch_size)):
                    batch = sentences[i: i + batch_size]
                    if self._workers:
                        rank = n % self.world_size
                        self._input_queues[rank].put((n, (batch, max_length, instruction)))
                        if n >= self.world_size:
                            _receive(self._output_queues[rank])
                    else:
                        result_dict[n] = self.model.embed(batch, max_length, instruction, self.device)
                        pbar.update(1)
        
        if self._workers:
            while len(result_dict) < num_batches:
                for oq in self._output_queues:
                    _receive(oq)

        pbar.close()
        results = [result_dict[n] for n in range(len(result_dict))]
        embeddings = torch.cat(results).float()
        assert embeddings.shape[0] == num_texts
        embeddings = embeddings.cpu().numpy()
        return embeddings

    def start(self):
        self.model.share_memory()
        logger.warning(f"Starting {self.world_size} worker processes.")

        mp_ctx = torch.multiprocessing.get_context('spawn')
        self._input_queues = [mp_ctx.Queue(self.mp_qsize) for _ in range(self.world_size)]
        self._output_queues = [mp_ctx.Queue(self.mp_qsize) for _ in range(self.world_size)]
        
        self._workers = list()
        for i, (iq, oq) in enumerate(zip(self._input_queues, self._output_queues)):
            device = torch.device(f'cuda:{i}')
            encode_worker = mp_ctx.Process(
                target=_encode_loop, name=f'encode_{i}', args=(
                    self.model, iq, oq, device, self.mp_qsize, self.amp_dtype
                )
            )
            encode_worker.start()
            self._workers.append(encode_worker)
            logger.warning(f"GPU {i} worker initiated.")

    def stop(self):
        [q.put(None) for q in self._input_queues]
        [w.join() for w in self._workers]
        [w.close() for w in self._workers]
        for qs in (self._input_queues, self._output_queues):
            [q.put(None) for q in qs]

model_name = "CoDiEmb/CoDi-MiniCPM"
model_name_or_path = "CoDiEmb/CoDi-MiniCPM"

codi_minicpm_instruction = {
    "CmedqaRetrieval": {"query": "<s>Instruction: Given a Chinese community medical question, retrieve replies that best answer the question \nQuery: ", "passage": "<s>"},
    "CovidRetrieval": {"query": "<s>Instruction: Given a question on COVID-19, retrieve news articles that answer the question \nQuery: ", "passage": "<s>"},
    "DuRetrieval": {"query": "<s>Instruction: Given a Chinese search query, retrieve web passages that answer the question \nQuery: ", "passage": "<s>"},
    "EcomRetrieval": {"query": "<s>Instruction: Given a user query from an e-commerce website, retrieve description sentences of relevant products \nQuery: ", "passage": "<s>"},
    "MedicalRetrieval": {"query": "<s>Instruction: Given a medical question, retrieve user replies that best answer the question \nQuery: ", "passage": "<s>"},
    "MMarcoRetrieval": {"query": "<s>Instruction: Given a web search query, retrieve relevant passages that answer the query \nQuery: ", "passage": "<s>"},
    "T2Retrieval": {"query": "<s>Instruction: Given a Chinese search query, retrieve web passages that answer the question \nQuery: ", "passage": "<s>"},
    "VideoRetrieval": {"query": "<s>Instruction: Given a video search query, retrieve the titles of relevant videos \nQuery: ", "passage": "<s>"},

    "AFQMC": "<s>Instruction: Represent the text in conversations between users and financial customer service, retrieve semantically similar text \nQuery: ",
    "ATEC": "<s>Instruction: Represent the text in conversations between users and financial customer service, retrieve semantically similar text \nQuery: ",
    "BQ": "<s>Instruction: Represent the user problem descriptions when handling bank credit business, retrieve semantically similar text \nQuery: ",
    "LCQMC": "<s>Instruction: Represent the user question descriptions on general question-answering platforms, retrieve semantically similar text \nQuery: ",
    "PAWSX": "<s>Instruction: Represent the Chinese Translations of English Encyclopedias, retrieve semantically similar text \nQuery: ",
    "QBQTC": "<s>Instruction: Represent the web search query, retrieve semantically similar text \nQuery: ",
    "STSB": "<s>Instruction: Represent the short general domain sentences, retrieve semantically similar text \nQuery: ",

    "T2Reranking": {"query": "<s>Instruction: Given a Chinese search query, retrieve web passages that answer the question \nQuery: ", "passage": "<s>"},
    "MMarcoReranking": {"query": "<s>Instruction: Given a web search query, retrieve relevant passages that answer the query \nQuery: ", "passage": "<s>"},
    "CMedQAv1-reranking": {"query": "<s>Instruction: Given a Chinese community medical question, retrieve replies that best answer the question \nQuery: ", "passage": "<s>"},
    "CMedQAv2-reranking": {"query": "<s>Instruction: Given a Chinese community medical question, retrieve replies that best answer the question \nQuery: ", "passage": "<s>"},

    "Ocnli": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "Cmnli": "<s>Instruction: Retrieve semantically similar text \nQuery: ",

    "TNews": "<s>Instruction: Classify the fine-grained category of the given news title \nQuery: ",
    "IFlyTek": "<s>Instruction: Given an App description text, find the appropriate fine-grained category \nQuery: ",
    "Waimai": "<s>Instruction: Classify the customer review from a food takeaway platform into positive or negative \nQuery: ",
    "OnlineShopping": "<s>Instruction: Classify the customer review for online shopping into positive or negative \nQuery: ",
    "JDReview": "<s>Instruction: Classify the customer review for iPhone on e-commerce platform into positive or negative \nQuery: ",
    "MultilingualSentiment": "<s>Instruction: Classify sentiment of the customer review into positive, neutral, or negative \nQuery: ",

    "CLSClusteringS2S": "<s>Instruction: Identify the main category of scholar papers based on the titles \nQuery: ",
    "CLSClusteringP2P": "<s>Instruction: Identify the main category of scholar papers based on the titles and abstracts \nQuery: ",
    "ThuNewsClusteringS2S": "<s>Instruction: Identify the topic or theme of the given news articles based on the titles \nQuery: ",
    "ThuNewsClusteringP2P": "<s>Instruction: Identify the topic or theme of the given news articles based on the titles and contents \nQuery: ",

    "ArguAna": "<s>Instruction: Given a claim, find documents that refute the claim \nQuery: ",
    "ClimateFEVER": "<s>Instruction: Given a claim about climate change, retrieve documents that support or refute the claim \nQuery: ",
    "ClimateFEVERHardNegatives": "<s>Instruction: Given a claim about climate change, retrieve documents that support or refute the claim \nQuery: ",
    "DBPedia": "<s>Instruction: Given a query, retrieve relevant entity descriptions from DBPedia \nQuery: ",
    "FEVER": "<s>Instruction: Given a claim, retrieve documents that support or refute the claim \nQuery: ",
    "FEVERHardNegatives": "<s>Instruction: Given a claim, retrieve documents that support or refute the claim \nQuery: ",
    "FiQA2018": "<s>Instruction: Given a financial question, retrieve user replies that best answer the question \nQuery: ",
    "HotpotQA": "<s>Instruction: Given a multi-hop question, retrieve documents that can help answer the question \nQuery: ",
    "HotpotQAHardNegatives": "<s>Instruction: Given a multi-hop question, retrieve documents that can help answer the question \nQuery: ",
    "MSMARCO": "<s>Instruction: Given a web search query, retrieve relevant passages that answer the query \nQuery: ",
    "NFCorpus": "<s>Instruction: Given a question, retrieve relevant documents that best answer the question \nQuery: ",
    "NQ": "<s>Instruction: Given a question, retrieve Wikipedia passages that answer the question \nQuery: ",
    "QuoraRetrieval": "<s>Instruction: Given a question, retrieve questions that are semantically equivalent to the given question \nQuery: ",
    "SCIDOCS": "<s>Instruction: Given a scientific paper title, retrieve paper abstracts that are cited by the given paper \nQuery: ",
    "SciFact": "<s>Instruction: Given a scientific claim, retrieve documents that support or refute the claim \nQuery: ",
    "Touche2020": "<s>Instruction: Given a question, retrieve detailed and persuasive arguments that answer the question \nQuery: ",
    "Touche2020Retrieval.v3": "<s>Instruction: Given a question, retrieve detailed and persuasive arguments that answer the question \nQuery: ",
    "TRECCOVID": "<s>Instruction: Given a query on COVID-19, retrieve documents that answer the query \nQuery: ",
	"CQADupstackRetrieval": "<s>Instruction: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question \nQuery: ",
    "CQADupstackGamingRetrieval": "<s>Instruction: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question \nQuery: ",
    "CQADupstackUnixRetrieval": "<s>Instruction: Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question \nQuery: ",
    
	"AskUbuntuDupQuestions": "<s>Instruction: Retrieve duplicate questions from AskUbuntu forum \nQuery: ",
    "MindSmallReranking": "<s>Instruction: Retrieve relevant news articles based on user browsing history \nQuery: ",

    "BIOSSES": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "STS16": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
	"SICK-R": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "STS13": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "STS14": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "STSBenchmark": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "FaroeseSTS": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "FinParaSTS": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "JSICK": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "IndicCrosslingualSTS": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "SemRel24STS": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "STS17": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "STS22.v2": "<s>Instruction: Retrieve semantically similar text \nQuery: ",
    "STSES": "<s>Instruction: Retrieve semantically similar text \nQuery: ",

    "SummEval": "<s>Instruction: Retrieve semantically similar text \nQuery: ",

    "ArXivHierarchicalClusteringP2P": "<s>Instruction: Identify the main and secondary category of Arxiv papers based on the titles and abstracts \nQuery: ",
    "ArXivHierarchicalClusteringS2S":  "Identify the main and secondary category of Arxiv papers based on the titles \nQuery: ",
	"ArxivClusteringP2P": "<s>Instruction: Identify the main and secondary category of Arxiv papers based on the titles and abstracts \nQuery: ",
    "ArxivClusteringS2S": "<s>Instruction: Identify the main and secondary category of Arxiv papers based on the titles \nQuery: ",
    "BiorxivClusteringP2P": "<s>Instruction: Identify the main category of Biorxiv papers based on the titles and abstracts \nQuery: ",
    "BiorxivClusteringS2S": "<s>Instruction: Identify the main category of Biorxiv papers based on the titles \nQuery: ",
    "MedrxivClusteringP2P": "<s>Instruction: Identify the main category of Medrxiv papers based on the titles and abstracts \nQuery: ",
    "MedrxivClusteringS2S": "<s>Instruction: Identify the main category of Medrxiv papers based on the titles \nQuery: ",
    "RedditClustering": "<s>Instruction: Identify the topic or theme of Reddit posts based on the titles \nQuery: ",
    "RedditClusteringP2P": "<s>Instruction: Identify the topic or theme of Reddit posts based on the titles and posts \nQuery: ",
    "StackExchangeClustering": "<s>Instruction: Identify the topic or theme of StackExchange posts based on the titles \nQuery: ",
    "StackExchangeClusteringP2P": "<s>Instruction: Identify the topic or theme of StackExchange posts based on the given paragraphs \nQuery: ",
    "TwentyNewsgroupsClustering": "<s>Instruction: Identify the topic or theme of the given news articles \nQuery: ",

	"SprintDuplicateQuestions": "<s>Instruction: Retrieve duplicate questions from Sprint forum \nQuery: ",
    "TwitterSemEval2015": "<s>Instruction: Retrieve tweets that are semantically similar to the given tweet \nQuery: ",
    "TwitterURLCorpus": "<s>Instruction: Retrieve tweets that are semantically similar to the given tweet \nQuery: ",

	"AmazonCounterfactualClassification": "<s>Instruction: Classify a given Amazon customer review text as either counterfactual or not-counterfactual \nQuery: ",
    "AmazonPolarityClassification": "<s>Instruction: Classify Amazon reviews into positive or negative sentiment \nQuery: ",
    "AmazonReviewsClassification": "<s>Instruction: Classify the given Amazon review into its appropriate rating category \nQuery: ",
    "Banking77Classification": "<s>Instruction: Given a online banking query, find the corresponding intents \nQuery: ",
    "EmotionClassification": "<s>Instruction: Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise \nQuery: ",
    "ImdbClassification": "<s>Instruction: Classify the sentiment expressed in the given movie review text from the IMDB dataset \nQuery: ",
    "MassiveIntentClassification": "<s>Instruction: Given a user utterance as query, find the user intents \nQuery: ",
    "MassiveScenarioClassification": "<s>Instruction: Given a user utterance as query, find the user scenarios \nQuery: ",
    "MTOPDomainClassification": "<s>Instruction: Classify the intent domain of the given utterance in task-oriented conversation \nQuery: ",
    "MTOPIntentClassification": "<s>Instruction: Classify the intent of the given utterance in task-oriented conversation \nQuery: ",
    "ToxicConversationsClassification": "<s>Instruction: Classify the given comments as either toxic or not toxic \nQuery: ",
    "TweetSentimentExtractionClassification": "<s>Instruction: Classify the sentiment of a given tweet as either positive, negative, or neutral \nQuery: "
}

training_data = {
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "T2Reranking": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "BQ": ["train"],
    "LCQMC": ["train"],
    "PAWSX": ["train"],
    "STS-B": ["train"],
    "AFQMC": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
}

CoDiEmb_MiniCPM = ModelMeta(
    name="CoDiEmb/CoDi-MiniCPM",
    languages=["zho-Hans"],
    revision="1652b960aae6a17e1466f219c3da0038baff2d2d",
    release_date="2025-08-18",
    loader=partial(
        LlmEmbedding,
        model_name_or_path,
        model_name=model_name,
        precision="bf16",
        max_length=1024,
        use_instruction=True,
        instruction_dict=codi_minicpm_instruction,
    ),
    open_weights=True,
    n_parameters=2724880896,
    memory_usage_mb=None,
    embed_dim=2304,
    license="apache-2.0",
    max_tokens=1024,
    reference="https://huggingface.co/CoDiEmb/CoDi-MiniCPM",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)