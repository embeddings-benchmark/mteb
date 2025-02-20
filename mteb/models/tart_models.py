import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from mteb.models.rerankers_monot5_based import ModelMeta, _loader
from typing import Any, Dict, List, Optional
from transformers import T5Tokenizer
from functools import partial
from mteb.models.rerankers_custom import RerankerWrapper, _loader

class EncT5ForSequenceClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config, dropout=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_output = hidden_states[:, 0, :]  # Take bos token (equiv. to <s>)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class EncT5Tokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=sp_model_kwargs,
            **kwargs,
        )

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return (
                [self.bos_token_id]
                + token_ids_0
                + [self.eos_token_id]
                + token_ids_1
                + [self.eos_token_id]
            )


    
class TARTFullReranker(RerankerWrapper):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        assert model_name_or_path in ["facebook/tart-full-flan-t5-xl", "facebook/tart-full-t0-3b"]
        self.model = EncT5ForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer =  EncT5Tokenizer.from_pretrained(model_name_or_path)
        self.max_length = 1024
        print(f"Using max_length of {self.max_length}")
        self.model.eval()


    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        queries, passages, instructions = list(zip(*input_to_rerank))
        if instructions is not None and instructions[0] is not None:
            # combine them with the queries with a [SEP] token
            if instructions[0].strip() == "": # empty instruction case, use generic
                queries = [f"{query} [SEP] Retrieve news paper paragraph to answer this question" for query in queries]
            else:
                queries = [f"{query} [SEP] {instruction}".strip() for query, instruction in zip(queries, instructions)]

        assert len(queries) == len(
            passages
        ), "queries and passages must be the same length"
        for query in queries:
            assert " [SEP] " in query, "query must contain [SEP]"

        if torch.cuda.is_available():
            self.model.to("cuda")
            # print("Loaded model to cuda")

        if self.first_print:
            print(f"Using {queries[0]}")
            self.first_print = False

        self.model.eval()

        features = self.tokenizer(
            queries,
            passages,
            padding=True,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
        )
        
        if torch.cuda.is_available():
            features = {k: v.to("cuda") for k, v in features.items()}
        with torch.no_grad():
            scores = self.model(**features).logits
            normalized_scores = [
                float(score[1]) for score in F.softmax(scores, dim=1)
            ]
        return normalized_scores
    

tart_full_flan_t5_xl = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=TARTFullReranker,
        model_name_or_path="facebook/tart-full-flan-t5-xl",
        fp_options="float16",
    ),
    name="facebook/tart-full-flan-t5-xl",
    languages=["eng_Latn"],
    open_weights=True,
    revision="9fedb1add4f922bbd578d6a82b0ee724cac5f804",
    release_date="2022-03-28",
    citation="""@article{asai2022tart,
  title={Task-aware Retrieval with Instructions},
  author={Asai, Akari and Schick, Timo and Lewis, Patrick and Chen, Xilun and Izacard, Gautier and Riedel, Sebastian and Hajishirzi, Hannaneh and Yih, Wen-tau},
  journal={arXiv preprint arXiv:2211.09260},
  year={2022}
}""",
)

tart_full_t0_3b = ModelMeta(
    loader=partial(  # type: ignore
        _loader,
        wrapper=TARTFullReranker,
        model_name_or_path="facebook/tart-full-t0-3b",
        fp_options="float16",
    ),
    name="facebook/tart-full-t0-3b",
    languages=["eng_Latn"],
    open_weights=True,
    revision="cc69254ccfd9aa5648b24e3eec548eb4e4c4deb2",
    release_date="2022-03-28",
    citation="""@article{asai2022tart,
  title={Task-aware Retrieval with Instructions},
  author={Asai, Akari and Schick, Timo and Lewis, Patrick and Chen, Xilun and Izacard, Gautier and Riedel, Sebastian and Hajishirzi, Hannaneh and Yih, Wen-tau},
  journal={arXiv preprint arXiv:2211.09260},
  year={2022}
}""",
)