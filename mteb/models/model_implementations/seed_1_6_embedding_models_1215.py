from __future__ import annotations

import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any

import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_implementations.bge_models import bge_chinese_training_data
from mteb.models.model_implementations.nvidia_models import nvidia_training_datasets
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

if TYPE_CHECKING:
    from PIL import Image


logger = logging.getLogger(__name__)

doubao_embedding_training_data = (
    {
        "PawsXPairClassification",
        "QBQTC",
        "STSB",
        "TNews",
        "Waimai",
        "IFlyTek",
        "MassiveScenarioClassification",
        "CodeSearchNetRetrieval",
    }
    | bge_chinese_training_data
    | nvidia_training_datasets
)


class Seed16EmbeddingWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        tokenizer_name: str = "cl100k_base",
        embed_dim: int | None = None,
        **kwargs,
    ) -> None:
        """Wrapper for Seed embedding API."""
        requires_package(
            self,
            "volcenginesdkarkruntime",
            "pip install mteb[ark]",
            "tiktoken",
        )

        self._model_name = model_name
        self._max_tokens = 32768
        self._embed_dim = embed_dim
        self._available_embed_dims = [2048, 1024]

    def pil_to_base64(self, image, format="jpeg"):
        if image is None:
            return None
        buffer = BytesIO()
        image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        encoded_bytes = base64.b64encode(img_bytes)
        return encoded_bytes.decode("utf-8")

    def multimodal_embedding(self, instruction, image_base64, text_content):
        auth_token = os.getenv("VOLCES_AUTH_TOKEN")
        model_name = "doubao-embedding-vision-251215"
        api_url = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "x-ark-vlm1": "true",
            "Content-Type": "application/json",
        }

        if text_content is not None and len(text_content) > self._max_tokens:
            text_content = text_content[: self._max_tokens]

        if image_base64 is not None and text_content is None:
            inputs = []
            for image in image_base64:
                image_format = "jpeg"
                image_data = f"data:image/{image_format};base64,{image}"
                inputs.append({"type": "image_url", "image_url": {"url": image_data}})

            payload = {"model": model_name, "input": inputs}
        elif image_base64 is None and text_content is not None:
            payload = {
                "model": model_name,
                "instruction": instruction,
                "input": [
                    {"type": "text", "text": text_content},
                ],
            }
        else:
            inputs = []
            for image in image_base64:
                image_format = "jpeg"
                image_data = f"data:image/{image_format};base64,{image}"
                inputs.append({"type": "image_url", "image_url": {"url": image_data}})
            inputs.append({"type": "text", "text": text_content})
            payload = {"model": model_name, "input": inputs}

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            response = requests.post(
                url=api_url, headers=headers, json=payload, timeout=30
            )

            if response.status_code != 200:
                retry_count += 1
                time.sleep(3)
                continue

            response_json = response.json()
            return response_json

        raise Exception(
            f"Request failed with status code {response.status_code}. "
            f"Response: {response.text}"
        )

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        assert (
            self._embed_dim is None or self._embed_dim in self._available_embed_dims
        ), (
            f"Available embed_dims are {self._available_embed_dims}, found {self._embed_dim}"
        )

        if images is not None and texts is not None:
            assert len(texts) == len(images)
            batch_len = len(texts)
            images_base64 = [self.pil_to_base64(image) for image in images]
        elif images is None:
            batch_len = len(texts)
            images_base64 = [None for _ in range(batch_len)]
        elif texts is None:
            batch_len = len(images)
            images_base64 = [self.pil_to_base64(image) for image in images]
        else:
            raise ValueError("images and texts cannot be None at the same time")

        def process_item(
            i, prompt_type, task_name, texts, images_base64, multimodal_embedding
        ):
            if (
                prompt_type == PromptType("query") or prompt_type is None
            ) and task_name in TASK_NAME_TO_INSTRUCTION:
                instruction = TASK_NAME_TO_INSTRUCTION[task_name]
                instruction = instruction.rstrip("{}").rstrip("\n")
                instruction = (
                    "Target_modality:Text.\n Instruction:" + instruction + "\n Query:"
                )
                input_text = texts[i]
            else:
                if texts[i] != "" and images_base64[i] is not None:
                    instruction = "Instruction: Compress the text and image into one word.\n Query:"
                    input_text = texts[i]
                elif texts[i] != "":
                    instruction = (
                        "Instruction: Compress the text into one word.\n Query:"
                    )
                    input_text = texts[i]
                elif images_base64[i] is not None:
                    instruction = (
                        "Instruction: Compress the image into one word.\n Query:"
                    )
                    input_text = None
                else:
                    raise ValueError("image and text are both None")

            resp = multimodal_embedding(
                instruction=instruction,
                image_base64=images_base64[i],
                text_content=input_text,
            )
            embedding = torch.tensor(resp["data"]["embedding"])
            embedding = torch.reshape(embedding, (1, -1))
            return embedding

        outputs = []
        process_partial = partial(
            process_item,
            prompt_type=prompt_type,
            task_name=task_name,
            texts=texts,
            images_base64=images_base64,
            multimodal_embedding=self.multimodal_embedding,
        )
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(process_partial, i) for i in range(batch_len)]
            for future in tqdm(futures, total=batch_len, desc="Encoding"):
                outputs.append(future.result())

        outputs = torch.stack(outputs, dim=0).squeeze(1)

        if self._embed_dim is not None:
            outputs = outputs[:, : self._embed_dim]
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return outputs.float()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if "text" in inputs.dataset.features:
            sentences = [text for batch in inputs for text in batch["text"]]
        else:
            sentences = None

        if "image" in inputs.dataset.features:
            images = [image for batch in inputs for image in batch["image"]]
        else:
            images = None

        return self.get_fused_embeddings(
            texts=sentences,
            images=images,
            task_name=task_metadata.name,
            prompt_type=prompt_type,
            **kwargs,
        )


TASK_NAME_TO_INSTRUCTION = {
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual\n{}",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment\n{}",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category\n{}",
    "Banking77Classification": "Given a online banking query, find the corresponding intents\n{}",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise\n{}",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset\n{}",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents\n{}",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios\n{}",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation\n{}",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation\n{}",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic\n{}",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral\n{}",
    "TNews": "Classify the fine-grained category of the given news title\n{}",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category\n{}",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative\n{}",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative\n{}",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative\n{}",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative\n{}",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts\n{}",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles\n{}",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts\n{}",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles\n{}",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts\n{}",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles\n{}",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles\n{}",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts\n{}",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles\n{}",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs\n{}",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles\n{}",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles\n{}",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts\n{}",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles\n{}",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents\n{}",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum\n{}",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history\n{}",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers\n{}",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum\n{}",
    "SprintDuplicateQuestions": "Retrieve semantically similar text\n{}",
    "TwitterSemEval2015": "Retrieve semantically similar text\n{}",
    "TwitterURLCorpus": "Retrieve semantically similar text\n{}",
    "T2Reranking": "为这个句子生成表示以用于检索相关内容：{}",
    "MmarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question\n{}",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question\n{}",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question\n{}",
    "Ocnli": "Retrieve semantically similar text\n{}",
    "Cmnli": "Retrieve semantically similar text\n{}",
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim\n{}",
    "ClimateFEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim\n{}",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia\n{}",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim\n{}",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim\n{}",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question\n{}",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question\n{}",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question\n{}",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query\n{}",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question\n{}",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question\n{}",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question\n{}",
    "SCIDOCS": "Given a title of a scientific paper, retrieve the titles of other relevant papers\n{}",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim\n{}",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question\n{}",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question\n{}",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query\n{}",
    "T2Retrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "MMarcoRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "DuRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "CovidRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "CmedqaRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "EcomRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "MedicalRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "VideoRetrieval": "为这个句子生成表示以用于检索相关内容：{}",
    "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text\n{}",
    "SICKFr": "Retrieve semantically similar text\n{}",
    "SummEvalFr": "Given a news summary, retrieve other semantically similar summaries\n{}",
    "MasakhaNEWSClassification": "Classify the News in the given texts into one of the seven category: politics,sports,health,business,entertainment,technology,religion \n{}",
    "OpusparcusPC": "Retrieve semantically similar text\n{}",
    "PawsX": "Retrieve semantically similar text\n{}",
    "AlloProfClusteringP2P": "Identify the main category of Allo Prof document based on the titles and descriptions\n{}",
    "AlloProfClusteringS2S": "Identify the topic of document titles from Allo Prof dataset\n{}",
    "HALClusteringS2S": "Identify the main category of academic passage based on the titles and contents\n{}",
    "MasakhaNEWSClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents\n{}",
    "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles\n{}",
    "MLSUMClusteringP2P": "Identify the topic or theme of the given articles based on the titles and contents\n{}",
    "MLSUMClusteringS2S": "Identify the topic or theme of the given articles based on the titles\n{}",
    "SyntecReranking": "Given a question, retrieve passages that answer the question\n{}",
    "AlloprofReranking": "Given a question, retrieve passages that answer the question\n{}",
    "AlloprofRetrieval": "Given a question, retrieve passages that answer the question\n{}",
    "BSARDRetrieval": "Given a question, retrieve passages that answer the question\n{}",
    "SyntecRetrieval": "Given a question, retrieve passages that answer the question\n{}",
    "XPQARetrieval": "Given a question, retrieve passages that answer the question\n{}",
    "MintakaRetrieval": "Given a question, retrieve passages that answer the question\n{}",
    "CBD": "Classify the sentiment of polish tweet reviews\n{}",
    "PolEmo2.0-IN": "Classify the sentiment of in-domain (medicine and hotels) online reviews\n{}",
    "PolEmo2.0-OUT": "Classify the sentiment of out-of-domain (products and school) online reviews\n{}",
    "AllegroReviews": "Classify the sentiment of reviews from e-commerce marketplace Allegro\n{}",
    "PAC": 'Classify the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"\n{}',
    "SICK-E-PL": "Retrieve semantically similar text\n{}",
    "SICK-R-PL": "Retrieve semantically similar text\n{}",
    "STS22": "Retrieve semantically similar text\n{}",
    "AFQMC": "Retrieve semantically similar text\n{}",
    "BQ": "Retrieve semantically similar text\n{}",
    "LCQMC": "Retrieve semantically similar text\n{}",
    "PAWSX": "Retrieve semantically similar text\n{}",
    "QBQTC": "Retrieve semantically similar text\n{}",
    "STS12": "Retrieve semantically similar text\n{}",
    "PPC": "Retrieve semantically similar text\n{}",
    "CDSC-E": "Retrieve semantically similar text\n{}",
    "PSC": "Retrieve semantically similar text\n{}",
    "8TagsClustering": "Identify of headlines from social media posts in Polish  into 8 categories: film, history, food, medicine, motorization, work, sport and technology\n{}",
    "ArguAna-PL": "Given a claim, find documents that refute the claim\n{}",
    "DBPedia-PL": "Given a query, retrieve relevant entity descriptions from DBPedia\n{}",
    "FiQA-PL": "Given a financial question, retrieve user replies that best answer the question\n{}",
    "HotpotQA-PL": "Given a multi-hop question, retrieve documents that can help answer the question\n{}",
    "MSMARCO-PL": "Given a web search query, retrieve relevant passages that answer the query\n{}",
    "NFCorpus-PL": "Given a question, retrieve relevant documents that best answer the question\n{}",
    "NQ-PL": "Given a question, retrieve Wikipedia passages that answer the question\n{}",
    "Quora-PL": "Given a question, retrieve questions that are semantically equivalent to the given question\n{}",
    "SCIDOCS-PL": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper\n{}",
    "SciFact-PL": "Given a scientific claim, retrieve documents that support or refute the claim\n{}",
    "TRECCOVID-PL": "Given a query on COVID-19, retrieve documents that answer the query\n{}",
    "GeoreviewClassification": "Classify the organization rating based on the reviews\n{}",
    "HeadlineClassification": "Classify the topic or theme of the given news headline\n{}",
    "InappropriatenessClassification": "Classify the given message as either sensitive topic or not\n{}",
    "KinopoiskClassification": "Classify the sentiment expressed in the given movie review text\n{}",
    "RuReviewsClassification": "Classify product reviews into positive, negative or neutral sentiment\n{}",
    "RuSciBenchGRNTIClassification": "Classify the category of scientific papers based on the titles and abstracts\n{}",
    "RuSciBenchOECDClassification": "Classify the category of scientific papers based on the titles and abstracts\n{}",
    "GeoreviewClusteringP2P": "Identify the organization category based on the reviews\n{}",
    "RuSciBenchGRNTIClusteringP2P": "Identify the category of scientific papers based on the titles and abstracts\n{}",
    "RuSciBenchOECDClusteringP2P": "Identify the category of scientific papers based on the titles and abstracts\n{}",
    "TERRa": "Given a premise, retrieve a hypothesis that is entailed by the premise\n{}",
    "RuBQReranking": "Given a question, retrieve Wikipedia passages that answer the question\n{}",
    "RiaNewsRetrieval": "Given a headline, retrieval relevant articles\n{}",
    "RuBQRetrieval": "Given a question, retrieve Wikipedia passages that answer the question\n{}",
    "RUParaPhraserSTS": "Retrieve semantically similar text\n{}",
    "RuSTSBenchmarkSTS": "Retrieve semantically similar text\n{}",
    "AppsRetrieval": "Given a code contest problem description, retrieve relevant code that can help solve the problem.\n{}",
    "COIRCodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.\n{}",
    "CodeEditSearchRetrieval": "Given a code commit message, retrieve the code difference information. \n{}",
    "CodeFeedbackMT": "Given a question about coding, retrieval code or passage that can solve user's question\n{}",
    "CodeFeedbackST": "Given a question about coding, retrieval code or passage that can solve user's question\n{}",
    "CodeSearchNetCCRetrieval": "Given a code comment, retrieve the code snippet corresponding to that comment.\n{}",
    "CodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.\n{}",
    "CodeTransOceanContest": "Given a piece for code, retrieval semantically similar code\n{}",
    "CodeTransOceanDL": "Given a piece for code, retrieval semantically similar code\n{}",
    "CosQA": "Given a question about coding, retrieval code or passage that can solve user's question\n{}",
    "StackOverflowQA": "Given a question about coding, retrieval code or passage that can solve user's question\n{}",
    "SyntheticText2SQL": "Given a user's question, retrieve SQL queries that are appropriate responses to the question\n{}",
    "BibleNLPBitextMining": "Retrieve parallel sentences\n{}",
    "BUCC.v2": "Retrieve parallel sentences\n{}",
    "DiaBlaBitextMining": "Retrieve parallel sentences\n{}",
    "FloresBitextMining": "Retrieve parallel sentences\n{}",
    "IN22GenBitextMining": "Retrieve parallel sentences\n{}",
    "IndicGenBenchFloresBitextMining": "Retrieve parallel sentences\n{}",
    "NollySentiBitextMining": "Retrieve parallel sentences\n{}",
    "NTREXBitextMining": "Retrieve parallel sentences\n{}",
    "NusaTranslationBitextMining": "Retrieve parallel sentences\n{}",
    "NusaXBitextMining": "Retrieve parallel sentences\n{}",
    "Tatoeba": "Retrieve parallel sentences\n{}",
    "BornholmBitextMining": "Retrieve parallel sentences\n{}",
    "NorwegianCourtsBitextMining": "Retrieve parallel sentences\n{}",
    "BulgarianStoreReviewSentimentClassfication": "Classify user reviews into positive or negative sentiment\n{}",
    "CzechProductReviewSentimentClassification": "Classify product reviews into positive or negative sentiment\n{}",
    "GreekLegalCodeClassification": "Given a greek legal text, classify its topic\n{}",
    "DBpediaClassification": "Given a Wikipedia articles, categorized it into classes based on its DBpedia ontology\n{}",
    "FinancialPhrasebankClassification": "Given financial news, categorized by sentiment into positive, negative, or neutral\n{}",
    "PoemSentimentClassification": "Gvien a poem, categorized by sentiment into positive, no_impact, negative or mixed\n{}",
    "TweetTopicSingleClassification": "Gvien a twitter, classify its topic\n{}",
    "EstonianValenceClassification": "Given a news article, categorized by sentiment into negatiivne, positiivne, neutraalne or vastuolulin\n{}",
    "FilipinoShopeeReviewsClassification": "Given a shop review, classify its rating on a scale from 1 to 5\n{}",
    "GujaratiNewsClassification": "Given a Gujarati news articles, classify its topic\n{}",
    "SentimentAnalysisHindi": "Given a hindi text, categorized by sentiment into positive, negative or neutral\n{}",
    "IndonesianIdClickbaitClassification": "Given an Indonesian news headlines, classify its into clickbait or non-clickbait\n{}",
    "ItaCaseholdClassification": "Given a judgments, classify its topic\n{}",
    "KorSarcasmClassification": "Given a twitter, categorized it into sarcasm or not_sarcasm\n{}",
    "KurdishSentimentClassification": "Given a text, categorized by sentiment into positive or negative\n{}",
    "MacedonianTweetSentimentClassification": "Given a Macedonian tweet, categorized by sentiment into positive, negative, or neutral\n{}",
    "AfriSentiClassification": "Given a text, categorized by sentiment into positive, negative, or neutral\n{}",
    "CataloniaTweetClassification": "Given a tweet, categorized by sentiment into AGAINST, FAVOR or NEUTRAL\n{}",
    "CyrillicTurkicLangClassification": "Given a text, classify its language\n{}",
    "IndicLangClassification": "Given a text, classify its language\n{}",
    "MultiHateClassification": "Given a text, categorized by sentiment into hate or non-hate\n{}",
    "NusaParagraphEmotionClassification": "Given a paragraph, classify its emotion\n{}",
    "NusaX-senti": "Given a text, categorized by sentiment into positive or negative\n{}",
    "SwissJudgementClassification": "Given a news article, categorized it into approval or dismissal\n{}",
    "NepaliNewsClassification": "Given a news article, categorized it into business, entertainment or sports\n{}",
    "OdiaNewsClassification": "Given a news article, categorized it into business, entertainment or sports\n{}",
    "PunjabiNewsClassification": "Given a news article, categorized it into two-classes\n{}",
    "SinhalaNewsClassification": "Given a news article, categorized it into political, business, technology, sports and Entertainment\n{}",
    "CSFDSKMovieReviewSentimentClassification": "Given a movie review, classify its rating on a scale from 0 to 5\n{}",
    "SiswatiNewsClassification": "Given a news article, classify its topic\n{}",
    "SlovakMovieReviewSentimentClassification": "Given a movie review, categorized it into positive or negative\n{}",
    "SwahiliNewsClassification": "Given a news article, classify its domain\n{}",
    "TswanaNewsClassification": "Given a news article, classify its topic\n{}",
    "IsiZuluNewsClassification": "Given a news article, classify its topic\n{}",
    "WikiCitiesClustering": "Identify of Wikipedia articles of cities by country\n{}",
    "RomaniBibleClustering": "Identify verses from the Bible in Kalderash Romani by book.\n{}",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts\n{}",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles\n{}",
    "BigPatentClustering.v2": "Identify the category of documents from the Big Patent dataset\n{}",
    "AlloProfClusteringS2S.v2": "Identify the topic of document titles from Allo Prof dataset\n{}",
    "HALClusteringS2S.v2": "Identify the topic of titles from HAL\n{}",
    "SIB200ClusteringS2S": "Identify the category of documents\n{}",
    "WikiClusteringP2P.v2": "Identify the category of wiki passages\n{}",
    "PlscClusteringP2P.v2": "Identify the category of titles+abstracts from Library of Science\n{}",
    "KorHateSpeechMLClassification": "Given a Korean online news comments, classify its fine-grained hate speech classes\n{}",
    "MalteseNewsClassification": "Given a maltese new, classify its topic\n{}",
    "MultiEURLEXMultilabelClassification": "Given a text, classify its topic\n{}",
    "BrazilianToxicTweetsClassification": "Given a tweet, classify its topic\n{}",
    "CTKFactsNLI": "Retrieve semantically similar text\n{}",
    "indonli": "Retrieve semantically similar text\n{}",
    "ArmenianParaphrasePC": "Retrieve semantically similar text\n{}",
    "PawsXPairClassification": "Retrieve semantically similar text\n{}",
    "RTE3": "Retrieve semantically similar text\n{}",
    "XNLI": "Retrieve semantically similar text\n{}",
    "PpcPC": "Retrieve semantically similar text\n{}",
    "GermanSTSBenchmark": "Retrieve semantically similar text\n{}",
    "SICK-R": "Retrieve semantically similar text\n{}",
    "STS13": "Retrieve semantically similar text\n{}",
    "STS14": "Retrieve semantically similar text\n{}",
    "STSBenchmark": "Retrieve semantically similar text\n{}",
    "FaroeseSTS": "Retrieve semantically similar text\n{}",
    "FinParaSTS": "Retrieve semantically similar text\n{}",
    "JSICK": "Retrieve semantically similar text\n{}",
    "IndicCrosslingualSTS": "Retrieve semantically similar text\n{}",
    "SemRel24STS": "Retrieve semantically similar text\n{}",
    "STS17": "Retrieve semantically similar text\n{}",
    "STS22.v2": "Retrieve semantically similar text\n{}",
    "STSES": "Retrieve semantically similar text\n{}",
    "STSB": "Retrieve semantically similar text\n{}",
    "AILAStatutes": "Identifying the most relevant statutes for a given situation\n{}",
    "HagridRetrieval": "Retrieval the relevant passage for the given query\n{}",
    "LegalBenchCorporateLobbying": "Retrieval the relevant passage for the given query\n{}",
    "LEMBPasskeyRetrieval": "Retrieval the relevant passage for the given query\n{}",
    "BelebeleRetrieval": "Retrieval the relevant passage for the given query\n{}",
    "MLQARetrieval": "Retrieval the relevant passage for the given query\n{}",
    "StatcanDialogueDatasetRetrieval": "Retrieval the relevant passage for the given query\n{}",
    "WikipediaRetrievalMultilingual": "Retrieval the relevant passage for the given query\n{}",
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query\n{}",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query\n{}",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query\n{}",
    "WebLINXCandidatesReranking": "Retrieval the relevant passage for the given query\n{}",
    "WikipediaRerankingMultilingual": "Retrieval the relevant passage for the given query\n{}",
    "STS15": "Retrieve semantically similar text\n{}",
    "MIRACLRetrievalHardNegatives": "Retrieval relevant passage for the given query\n{}",
    "BIOSSES": "Retrieve semantically similar text\n{}",
    "CQADupstackRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\n{}",
    "STS16": "Retrieve semantically similar text\n{}",
    "SummEval": "Retrieve semantically similar text\n{}",
    "ATEC": "Retrieve semantically similar text\n{}",
    "ArguAna": "Given a claim, find documents that refute the claim\n{}",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\n{}",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question\n{}",
    "MMarcoReranking": "为这个句子生成表示以用于检索相关内容：{}",
    "CMedQAv1-reranking": "为这个句子生成表示以用于检索相关内容：{}",
    "CMedQAv2-reranking": "为这个句子生成表示以用于检索相关内容：{}",
    "SummEvalSummarization.v2": "Retrieve semantically similar text\n{}",
    "BiorxivClusteringP2P.v2": "Identify the main category of Biorxiv papers based on the titles and abstracts\n{}",
    "MedrxivClusteringP2P.v2": "Identify the main category of Medrxiv papers based on the titles and abstracts\n{}",
    "MedrxivClusteringS2S.v2": "Identify the main category of Medrxiv papers based on the titles\n{}",
    "StackExchangeClustering.v2": "Identify the topic or theme of StackExchange posts based on the titles\n{}",
    "StackExchangeClusteringP2P.v2": "Identify the topic or theme of StackExchange posts based on the given paragraphs\n{}",
    "TwentyNewsgroupsClustering.v2": "Identify the topic or theme of the given news articles\n{}",
    "SwednClusteringP2P": "Identify news categories in Swedish passages\n{}",
    "CEDRClassification": "Given a comment as query, find expressed emotions (joy, sadness, surprise, fear, and anger)\n{}",
    "TwitterHjerneRetrieval": "Retrieve answers to questions asked in Danish tweets.\n{}",
    "TempReasonL1": "Given the following question about time, retrieve the correct answer.\n{}",
    "WinoGrande": "Given the following sentence, retrieve an appropriate answer to fill in the missing underscored part.\n{}",
    "NordicLangClassification": "Classify texts based on language\n{}",
    "CLSClusteringP2P.v2": "Identify the topic or theme of the given news articles based on the titles\n{}",
    "ScalaClassification": "Classify passages in Scandinavian Languages based on linguistic acceptability\n{}",
    "SpartQA": "Given the following spatial reasoning question, retrieve the right answer.\n{}",
    "DalajClassification": "Classify texts based on linguistic acceptability in Swedish\n{}",
    "VoyageMMarcoReranking": "Given a Japanese search query, retrieve web passages that answer the question\n{}",
}


seed_embedding = ModelMeta(
    name="Bytedance/Seed1.6-embedding-1215",
    revision="1",
    release_date="2025-12-15",
    languages=[
        "afr-Latn",
        "ara-Arab",
        "aze-Latn",
        "bel-Cyrl",
        "bul-Cyrl",
        "ben-Beng",
        "cat-Latn",
        "ceb-Latn",
        "ces-Latn",
        "cym-Latn",
        "dan-Latn",
        "deu-Latn",
        "ell-Grek",
        "eng-Latn",
        "spa-Latn",
        "est-Latn",
        "eus-Latn",
        "fas-Arab",
        "fin-Latn",
        "fra-Latn",
        "glg-Latn",
        "guj-Gujr",
        "heb-Hebr",
        "hin-Deva",
        "hrv-Latn",
        "hat-Latn",
        "hun-Latn",
        "hye-Armn",
        "ind-Latn",
        "isl-Latn",
        "ita-Latn",
        "jpn-Jpan",
        "jav-Latn",
        "kat-Geor",
        "kaz-Cyrl",
        "khm-Khmr",
        "kan-Knda",
        "kor-Hang",
        "kir-Cyrl",
        "lao-Laoo",
        "lit-Latn",
        "lav-Latn",
        "mkd-Cyrl",
        "mal-Mlym",
        "mon-Cyrl",
        "mar-Deva",
        "msa-Latn",
        "mya-Mymr",
        "nep-Deva",
        "nld-Latn",
        "nor-Latn",
        "nob-Latn",
        "nno-Latn",
        "pan-Guru",
        "pol-Latn",
        "por-Latn",
        "que-Latn",
        "ron-Latn",
        "rus-Cyrl",
        "sin-Sinh",
        "slk-Latn",
        "slv-Latn",
        "swa-Latn",
        "tam-Taml",
        "tel-Telu",
        "tha-Thai",
        "tgl-Latn",
        "tur-Latn",
        "ukr-Cyrl",
        "urd-Arab",
        "vie-Latn",
        "yor-Latn",
        "zho-Hans",
    ],
    loader=Seed16EmbeddingWrapper,
    max_tokens=32768,
    embed_dim=2048,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license=None,
    reference="https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-embedding-vision",
    similarity_fn_name="cosine",
    framework=["API"],
    use_instructions=True,
    training_datasets=doubao_embedding_training_data,
    public_training_code=None,
    public_training_data=None,
)
