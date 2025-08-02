import mteb
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType
from functools import partial
from typing import Dict, List, Optional, Union
from tqdm.autonotebook import trange

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def get_detailed_instruct(instruction: str) -> str:
    if not instruction: return ''

    return 'Instruct: {}\nQuery: '.format(instruction)

def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

task_to_instructions = {
        "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
        "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
        "ArXivHierarchicalClusteringS2S":  "Identify the main and secondary category of Arxiv papers based on the titles",
        "ArguAna": "Given a claim, find documents that refute the claim",
        "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
        "BIOSSES": "Retrieve semantically similar text",
        "Banking77Classification": "Given a online banking query, find the corresponding intents",
        "BiorxivClusteringP2P.v2": "Identify the main category of Biorxiv papers based on the titles and abstracts",
        "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "ClimateFEVERHardNegatives": "Given a claim about climate change, retrieve documents that support or refute the claim",
        "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim",
        "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
        "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question",
        "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
        "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
        "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
        "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
        "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
        "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
        "SICK-R": "Retrieve semantically similar text",
        "STS12": "Retrieve semantically similar text",
        "STS13": "Retrieve semantically similar text",
        "STS14": "Retrieve semantically similar text",
        "STS15": "Retrieve semantically similar text",
        "STS17": "Retrieve semantically similar text",
        "STS22.v2": "Retrieve semantically similar text",
        "STSBenchmark": "Retrieve semantically similar text",
        "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
        "StackExchangeClustering.v2": "Identify the topic or theme of StackExchange posts based on the titles",
        "StackExchangeClusteringP2P.v2": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
        "SummEvalSummarization.v2": "Retrieve semantically similar text",
        "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
        "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
        "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic",
        "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
        "TwentyNewsgroupsClustering.v2": "Identify the topic or theme of the given news articles",
        "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
        "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
        'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
        'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
        'CMedQAv1-reranking': 'Given a Chinese community medical question, retrieve replies that best answer the question',
        'CMedQAv2-reranking': 'Given a Chinese community medical question, retrieve replies that best answer the question',
        'ATEC': 'Retrieve semantically similar text',
        'BQ': 'Retrieve semantically similar text',
        'LCQMC': 'Retrieve semantically similar text',
        'PAWSX': 'Retrieve semantically similar text',
        'STSB': 'Retrieve semantically similar text',
        'AFQMC': 'Retrieve semantically similar text',
        'QBQTC': 'Retrieve semantically similar text',
        'TNews': 'Classify the fine-grained category of the given news title',
        'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
        'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
        'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
        'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
        'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
        'Ocnli': 'Retrieve semantically similar text.',
        'Cmnli': 'Retrieve semantically similar text.',
        'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
        'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
        'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
        'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
        'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
        'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
        'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
        'MMarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
        'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
        'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
        'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
        'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos'
        }

class QZhouModel(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_length: int = 8192
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        model = AutoModel.from_pretrained(model_name_or_path, device_map="cuda", trust_remote_code=True, torch_dtype=torch.bfloat16)
      
        return cls(model=model, tokenizer=tokenizer)

    def tokenize(self, sents, prompt):
        
        sents = [prompt + x for x in sents]
        model_inputs = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return model_inputs


    def forward(self, sentence_feature: Dict[str, Tensor]):
        reps = self.model(**sentence_feature)
        
        return F.normalize(self.get_pooling(sentence_feature, reps.last_hidden_state), p=2, dim=-1)

    def get_pooling(self, features, last_hidden_states):  # Mean_Pooling 
        seq_lengths = features["attention_mask"].sum(dim=-1)

        return torch.stack(
            [
                last_hidden_states[i, -length:, :].sum(dim=0) / length
                for i, length in enumerate(seq_lengths)
            ],
            dim=0,
        )

    def encode(self, 
                sentences: Union[str, List[str]],
                **kwargs):
        
        return self._encode(sentences, **kwargs)

    
    @torch.no_grad()
    def _encode(
        self,
        sentences: Union[str, List[str]],
        prompt: str = '',
        batch_size: int = 32,
        show_progress_bar: bool = True,
        device: Optional[str] = None,
    ):
        batch_size = 256

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        for start_index in trange(
            0,
            len(sentences),
            batch_size,
            desc=f"Batches_{device}",
            disable=not show_progress_bar,
            position=int(device[-1]) if device[-1].isdigit() else 0
        ):
            sentences_batch = sentences_sorted[
                start_index : start_index + batch_size
            ]
            
            sentences_batch = [(x if x != '' else (x + 'Null')) for x in sentences_batch]
            features = self.tokenize(sentences_batch, prompt)
            features = batch_to_device(features, device)
            with torch.no_grad():
                embeddings = self.forward(features)
                embeddings = embeddings.detach()
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
 
        return len(text)

query_passage_instruction = ['AmazonCounterfactualClassification', 'ArXivHierarchicalClusteringP2P', 'ArXivHierarchicalClusteringS2S', \
    'BIOSSES', 'Banking77Classification', 'BiorxivClusteringP2P.v2', 'ImdbClassification', 'MTOPDomainClassification', \
    'MassiveIntentClassification', 'MassiveScenarioClassification', 'MedrxivClusteringP2P', 'MedrxivClusteringS2S', \
    'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS17', 'STS22.v2', 'STSBenchmark', 'SprintDuplicateQuestions', \
    'StackExchangeClustering.v2', 'StackExchangeClusteringP2P.v2', 'ToxicConversationsClassification', \
    'TweetSentimentExtractionClassification', 'TwentyNewsgroupsClustering.v2', 'TwitterSemEval2015', 'TwitterURLCorpus', 'AskUbuntuDupQuestions', 'MindSmallReranking', \
    'CQADupstackGamingRetrieval', 'CQADupstackUnixRetrieval', 'ArguAna', 'SummEvalSummarization.v2', "AFQMC", "ATEC", "BQ", "LCQMC", 'STSB', \
    "QBQTC", 'PAWSX', 'CLSClusteringP2P', 'CLSClusteringS2S', 'ThuNewsClusteringS2S', 'ThuNewsClusteringP2P', 'Cmnli', \
    'Ocnli', 'IFlyTek', 'JDReview', 'MultilingualSentiment', 'OnlineShopping', 'Waimai', 'TNews', 'MMarcoReranking', \
    'CMedQAv1-reranking', 'CMedQAv2-reranking', 'T2Reranking']


class QZhouModelWrapper(Wrapper):
    def __init__(self, model_name, model_revision):
        # super().__init__(model_name, model_revision)
        self.model = QZhouModel.from_pretrained(model_name)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        if prompt_type == PromptType.query or task_name in query_passage_instruction:
            instruction = get_detailed_instruct(task_to_instructions[task_name])
        else:
            instruction = ''
   
        return self.model.encode(sentences=sentences, prompt=instruction, **kwargs)


QZhou_Embedding = ModelMeta(
    loader = partial(
        QZhouModelWrapper,
        model_name="Kingsoft-LLM/QZhou-Embedding",
        model_revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
    ),
    name="Kingsoft-LLM/QZhou-Embedding",
    languages=["eng-Latn", "zho-Hans"], 
    open_weights=True,
    revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
    release_date="2025-08-01",
    n_parameters=7_070_619_136,
    memory_usage_mb=29070,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Kingsoft-LLM/QZhou-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/cfli/datasets",
    training_datasets={"bge-e5data": ["train"], "bge-full-data": ['train']},
)

# if __name__ == '__main__':
#     loader = partial(
#         QZhouModelWrapper,
#         model_name="Kingsoft-LLM/QZhou-Embedding",
#         model_revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a"
#     )()
#     for task in ['AFQMC', 'STS12']:
#         tasks = mteb.get_tasks(tasks=[task])
#         evaluation = mteb.MTEB(tasks=tasks)
#         evaluation.run(loader, output_folder="results")