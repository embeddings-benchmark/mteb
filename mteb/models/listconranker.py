from mteb.model_meta import ModelMeta
from mteb.models.rerankers_custom import RerankerWrapper
from torch.utils.data import DataLoader

import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from typing import Union, List
from mteb.abstasks import TaskMetadata
from mteb.model_meta import ModelMeta
from mteb.types import BatchedInput, PromptType

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from torch import nn
from torch.nn import functional as F
from typing import Any


'''
A example to reproduce the results of ListConRanker

import mteb
import sys

model = mteb.get_model('ByteDance/ListConRanker', listconranker_local_path=YOUR_LOCAL_MODEL_PATH)
tasks = mteb.get_tasks(tasks=['T2Reranking], languages=['zho-Hans'])
evaluation = mteb.MTEB(tasks=tasks)
encode_kwargs = {'batch_size': sys.maxsize}
evaluation.run(model, encode_kwargs=encode_kwargs, previous_results=YOUR_LOCAL_PREVIOUS_RESULTS_PATH)
'''


class ListTransformer(nn.Module):
    def __init__(self, num_layer, config, device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.list_transformer_layer = nn.TransformerEncoderLayer(1792, self.config.num_attention_heads, batch_first=True, activation=F.gelu, norm_first=False)
        self.list_transformer = nn.TransformerEncoder(self.list_transformer_layer, num_layer)
        self.relu = nn.ReLU()
        self.query_embedding = QueryEmbedding(config, device)

        self.linear_score3 = nn.Linear(1792 * 2, 1792)
        self.linear_score2 = nn.Linear(1792 * 2, 1792)
        self.linear_score1 = nn.Linear(1792 * 2, 1)

    def forward(self, pair_features, pair_nums):
        pair_nums = [x + 1 for x in pair_nums]
        batch_pair_features = pair_features.split(pair_nums)

        pair_feature_query_passage_concat_list = []
        for i in range(len(batch_pair_features)):
            pair_feature_query = batch_pair_features[i][0].unsqueeze(0).repeat(pair_nums[i] - 1, 1)
            pair_feature_passage = batch_pair_features[i][1:]
            pair_feature_query_passage_concat_list.append(torch.cat([pair_feature_query, pair_feature_passage], dim=1))
        pair_feature_query_passage_concat = torch.cat(pair_feature_query_passage_concat_list, dim=0)

        batch_pair_features = nn.utils.rnn.pad_sequence(batch_pair_features, batch_first=True)

        query_embedding_tags = torch.zeros(batch_pair_features.size(0), batch_pair_features.size(1), dtype=torch.long, device=self.device)
        query_embedding_tags[:, 0] = 1
        batch_pair_features = self.query_embedding(batch_pair_features, query_embedding_tags)

        mask = self.generate_attention_mask(pair_nums)
        query_mask = self.generate_attention_mask_custom(pair_nums)
        pair_list_features = self.list_transformer(batch_pair_features, src_key_padding_mask=mask, mask=query_mask)

        output_pair_list_features = []
        output_query_list_features = []
        pair_features_after_transformer_list = []
        for idx, pair_num in enumerate(pair_nums):
            output_pair_list_features.append(pair_list_features[idx, 1:pair_num, :])
            output_query_list_features.append(pair_list_features[idx, 0, :])
            pair_features_after_transformer_list.append(pair_list_features[idx, :pair_num, :])

        pair_features_after_transformer_cat_query_list = []
        for idx, pair_num in enumerate(pair_nums):
            query_ft = output_query_list_features[idx].unsqueeze(0).repeat(pair_num - 1, 1)
            pair_features_after_transformer_cat_query = torch.cat([query_ft, output_pair_list_features[idx]], dim=1)
            pair_features_after_transformer_cat_query_list.append(pair_features_after_transformer_cat_query)
        pair_features_after_transformer_cat_query = torch.cat(pair_features_after_transformer_cat_query_list, dim=0)

        pair_feature_query_passage_concat = self.relu(self.linear_score2(pair_feature_query_passage_concat))
        pair_features_after_transformer_cat_query = self.relu(self.linear_score3(pair_features_after_transformer_cat_query))
        final_ft = torch.cat([pair_feature_query_passage_concat, pair_features_after_transformer_cat_query], dim=1)
        logits = self.linear_score1(final_ft).squeeze()

        return logits, torch.cat(pair_features_after_transformer_list, dim=0)

    def generate_attention_mask(self, pair_num):
        max_len = max(pair_num)
        batch_size = len(pair_num)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        for i, length in enumerate(pair_num):
            mask[i, length:] = True
        return mask

    def generate_attention_mask_custom(self, pair_num):
        max_len = max(pair_num)

        mask = torch.zeros(max_len, max_len, dtype=torch.bool, device=self.device)
        mask[0, 1:] = True

        return mask


class QueryEmbedding(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()
        self.query_embedding = nn.Embedding(2, 1792)
        self.layerNorm = nn.LayerNorm(1792)

    def forward(self, x, tags):
        query_embeddings = self.query_embedding(tags)
        x += query_embeddings
        x = self.layerNorm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, list_transformer_layer_4eval: int=None):
        super().__init__()
        self.hf_model = hf_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()

        self.config = self.hf_model.config
        self.config.output_hidden_states = True

        self.linear_in_embedding = nn.Linear(1024, 1792)
        self.list_transformer_layer = list_transformer_layer_4eval
        self.list_transformer = ListTransformer(self.list_transformer_layer, self.config, self.device)

    def forward(self, batch):
        if 'pair_num' in batch:
            pair_nums = batch.pop('pair_num').tolist()

        if self.training:
            pass
        else:
            split_batch = 400
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            if sum(pair_nums) > split_batch:
                last_hidden_state_list = []
                input_ids_list = input_ids.split(split_batch)
                attention_mask_list = attention_mask.split(split_batch)
                for i in range(len(input_ids_list)):
                    last_hidden_state = self.hf_model(input_ids=input_ids_list[i], attention_mask=attention_mask_list[i], return_dict=True).hidden_states[-1]
                    last_hidden_state_list.append(last_hidden_state)
                last_hidden_state = torch.cat(last_hidden_state_list, dim=0)
            else:
                ranker_out = self.hf_model(**batch, return_dict=True)
                last_hidden_state = ranker_out.last_hidden_state

            pair_features = self.average_pooling(last_hidden_state, attention_mask)
            pair_features = self.linear_in_embedding(pair_features)

            logits, pair_features_after_list_transformer = self.list_transformer(pair_features, pair_nums)
            logits = self.sigmoid(logits)

            return logits

    @classmethod
    def from_pretrained_for_eval(cls, model_name_or_path, list_transformer_layer):
        hf_model = AutoModel.from_pretrained(model_name_or_path)
        reranker = cls(hf_model, list_transformer_layer)
        reranker.linear_in_embedding.load_state_dict(torch.load(model_name_or_path + '/linear_in_embedding.pt'))
        reranker.list_transformer.load_state_dict(torch.load(model_name_or_path + '/list_transformer.pt'))
        return reranker

    def average_pooling(self, hidden_state, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).to(dtype=hidden_state.dtype)
        masked_hidden_state = hidden_state * extended_attention_mask
        sum_embeddings = torch.sum(masked_hidden_state, dim=1)
        sum_mask = extended_attention_mask.sum(dim=1)
        return sum_embeddings / sum_mask


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ListConRanker(RerankerWrapper):
    def __init__(
            self,
            model_name_or_path: str = None,
            **kwargs
    ) -> None:
        super().__init__(model_name_or_path, **kwargs)
        if 'listconranker_local_path' not in kwargs:
            raise ValueError('Please specify the local path of the model through listconranker_local_path')
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['listconranker_local_path'])
        self.model = Encoder.from_pretrained_for_eval(kwargs['listconranker_local_path'], list_transformer_layer=2)
        
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options

        self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def compute_score(self, query_passages_list: List[List[str]], max_length: int = 512) -> List[List[float]]:
        pair_nums = [len(pairs) - 1 for pairs in query_passages_list]
        sentences_batch = sum(query_passages_list, [])
        inputs = self.tokenizer(
            sentences_batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length,
        ).to(self.device)
        inputs['pair_num'] = torch.LongTensor(pair_nums)
        scores = self.model(inputs).float()
        all_scores = scores.cpu().numpy().tolist()

        if isinstance(all_scores, float):
            return [all_scores]
        result = []
        curr_idx = 0
        for i in range(len(pair_nums)):
            result.append(all_scores[curr_idx: curr_idx + pair_nums[i]])
            curr_idx += pair_nums[i]
        # return all_scores
        return result

    @torch.no_grad()
    def iterative_inference(self, query_passages: List[str], max_length: int = 512) -> List[float]:
        query = query_passages[0]
        passage = query_passages[1:]

        filter_times = 0
        passage2score = {}
        while len(passage) > 20:
            batch = [[query] + passage]
            pred_scores = self.compute_score(batch, max_length)[0]
             # Sort in increasing order
            pred_scores_argsort = np.argsort(pred_scores).tolist()
            passage_len = len(passage)
            to_filter_num = math.ceil(passage_len * 0.2)
            if to_filter_num < 10:
                to_filter_num = 10

            have_filter_num = 0
            while have_filter_num < to_filter_num:
                idx = pred_scores_argsort[have_filter_num]
                if passage[idx] in passage2score:
                    passage2score[passage[idx]].append(pred_scores[idx] + filter_times)
                else:
                    passage2score[passage[idx]] = [pred_scores[idx] + filter_times]
                have_filter_num += 1
            while pred_scores[pred_scores_argsort[have_filter_num - 1]] == pred_scores[pred_scores_argsort[have_filter_num]]:
                idx = pred_scores_argsort[have_filter_num]
                if passage[idx] in passage2score:
                    passage2score[passage[idx]].append(pred_scores[idx] + filter_times)
                else:
                    passage2score[passage[idx]] = [pred_scores[idx] + filter_times]
                have_filter_num += 1
            next_passage = []
            next_passage_idx = have_filter_num
            while next_passage_idx < len(passage):
                idx = pred_scores_argsort[next_passage_idx]
                next_passage.append(passage[idx])
                next_passage_idx += 1
            passage = next_passage
            filter_times += 1

        batch = [[query] + passage]
        pred_scores = self.compute_score(batch, max_length)[0]
        cnt = 0
        while cnt < len(passage):
            if passage[cnt] in passage2score:
                passage2score[passage[cnt]].append(pred_scores[cnt] + filter_times)
            else:
                passage2score[passage[cnt]] = [pred_scores[cnt] + filter_times]
            cnt += 1

        passage = query_passages[1:]
        final_score = []
        for i in range(len(passage)):
            p = passage[i]
            final_score.append(passage2score[p][0])
        return final_score
    
    @torch.inference_mode()
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        
        instructions = None
        queries = [text for batch in inputs1 for text in batch["query"]]
        passages = [text for batch in inputs2 for text in batch["text"]['text']]
        
        assert len(queries) == len(passages)

        final_scores = []
        query = queries[0]
        tmp_passages = []
        if 'traditional_inference' in kwargs and kwargs['traditional_inference']:
            for q, p in zip(queries, passages):
                if query == q:
                    tmp_passages.append(p)
                else:
                    query_passages_tuples = [[query] + tmp_passages]
                    scores = self.compute_score(query_passages_tuples)[0]
                    final_scores += scores

                    query = q
                    tmp_passages = [p]
            if len(tmp_passages) > 0:
                query_passages_tuples = [[query] + tmp_passages]
                scores = self.compute_score(query_passages_tuples)[0]
                final_scores += scores
        else:
            for q, p in zip(queries, passages):
                if query == q:
                    tmp_passages.append(p)
                else:
                    query_passages = [query] + tmp_passages
                    scores = self.iterative_inference(query_passages)
                    final_scores += scores

                    query = q
                    tmp_passages = [p]
            if len(tmp_passages) > 0:
                query_passages = [query] + tmp_passages
                scores = self.iterative_inference(query_passages)
                final_scores += scores

        assert len(final_scores) == len(queries), (
            f"Expected {len(queries)} scores, got {len(final_scores)}"
        )
        
        return final_scores
        

listconranker = ModelMeta(
    loader=ListConRanker,
    loader_kwargs=dict(
        fp_options="float16",
    ),
    name='ByteDance/ListConRanker',
    languages=['zho-Hans'],
    open_weights=True,
    revision='2852d459b0e3942733da8756b809f3659649d5a6',
    release_date='2024-12-11',
    n_parameters=401_000_000,
    memory_usage_mb=None,
    similarity_fn_name=None,
    training_datasets=None,
    embed_dim=1024,
    license='mit',
    max_tokens=512,
    reference='https://huggingface.co/ByteDance/ListConRanker',
    framework=['PyTorch'],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    is_cross_encoder=True
)