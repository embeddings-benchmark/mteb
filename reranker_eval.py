import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from mteb import MTEB
from mteb.evaluation.evaluators.RetrievalEvaluator import Reranker

prediction_tokens = {
    "castorini/monot5-small-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-small-msmarco-100k": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco": ["▁false", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
}


class MonoT5Reranker(Reranker):
    name: str = "MonoT5"
    prompt_template: str = "Query: {query} Document: {text} Relevant:"

    def __init__(
        self,
        model_name_or_path="castorini/monot5-base-msmarco-10k",
        **kwargs,
    ):
        self.device = None
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if "fp_options" in kwargs:
            model_args["torch_dtype"] = kwargs["fp_options"]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, **model_args
        )
        print(f"Using model {model_name_or_path}")

        if "torch_compile" in kwargs and kwargs["torch_compile"]:
            self.torch_compile = kwargs["torch_compile"]
            self.model = torch.compile(self.model)
        else:
            self.torch_compile = False

        self.first_print = True

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            model_name_or_path,
            self.tokenizer,
            kwargs["token_false"] if "token_false" in kwargs else None,
            kwargs["token_true"] if "token_true" in kwargs else None,
        )
        print(f"Using max_length of {self.tokenizer.model_max_length}")
        print(f"Using token_false_id of {self.token_false_id}")
        print(f"Using token_true_id of {self.token_true_id}")
        self.max_length = self.tokenizer.model_max_length
        print(f"Using max_length of {self.max_length}")

        self.model.eval()

    def get_prediction_tokens(
        self, model_name_or_path, tokenizer, token_false=None, token_true=None
    ):
        if not (token_false and token_true):
            if model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[model_name_or_path]
                token_false_id = tokenizer.get_vocab()[token_false]
                token_true_id = tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                return self.get_prediction_tokens(
                    "castorini/monot5-base-msmarco", self.tokenizer
                )
        else:
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id

    @torch.inference_mode()
    def predict(self, inputs):
        queries, passages, instructions = zip(*inputs)
        if instructions is not None and instructions[0] is not None:
            # print(f"Adding instructions to monot5 queries")
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]

        prompts = [
            self.prompt_template.format(query=query, text=text)
            for (query, text) in zip(queries, passages)
        ]
        if self.first_print:
            print(f"Using {prompts[0]}")
            self.first_print = False

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            pad_to_multiple_of=(8 if self.torch_compile else None),
        ).to(self.device)
        output = self.model.generate(
            **tokens,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        batch_scores = output.scores[0]
        batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()


# model = MonoT5Reranker("castorini/monot5-base-msmarco-10k")
model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
first_stage_model = SentenceTransformer("all-MiniLM-L6-v2")
for task in ["NFCorpus", "News21InstructionRetrieval"]:
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = MTEB(
        tasks=[task], task_langs=["en"]
    )  # Remove "en" for running all languages
    evaluation.run(
        model, eval_splits=eval_splits, top_k=10, first_stage=first_stage_model
    )
