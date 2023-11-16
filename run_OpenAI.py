"""
openai==0.26.4
tiktoken==0.2.0
"""
import argparse
import logging
import os
import pathlib
import pickle

from mteb.evaluation.MTEB import MTEB
import openai
import tiktoken

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="1" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/gpfswork/rech/six/commun/models"
os.environ["HF_DATASETS_CACHE"]="/gpfswork/rech/six/commun/datasets"
os.environ["HF_MODULES_CACHE"]="/gpfswork/rech/six/commun/modules"
os.environ["HF_METRICS_CACHE"]="/gpfswork/rech/six/commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OpenAIEmbedder:
    """
    Benchmark OpenAIs embeddings endpoint.
    """
    def __init__(self, engine, task_name=None, batch_size=32, save_emb=False, **kwargs):
        self.engine = engine
        self.max_token_len = 8191
        self.batch_size = batch_size
        self.save_emb = save_emb # Problematic as the filenames may end up being the same
        self.base_path = f"embeddings/{engine.split('/')[-1]}/"
        # self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer = tiktoken.encoding_for_model(engine)
        self.task_name = task_name

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
    def encode(self, 
            sentences,
            decode=True,
            idx=None,
            **kwargs
        ):

        openai.api_key = API_KEY

        fin_embeddings = []

        embedding_path = f"{self.base_path}/{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}.pickle"
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]

                all_tokens = []
                used_indices = []
                for j, txt in enumerate(batch):
                    # tokens = self.tokenizer.encode(txt, add_special_tokens=False)
                    if not(txt):
                        print("Detected empty item, which is not allowed by the OpenAI API - Replacing with empty space")
                        txt = " "
                    tokens = self.tokenizer.encode(txt)
                    token_len = len(tokens)
                    if token_len > self.max_token_len:
                        tokens = tokens[:self.max_token_len]
                    # For some characters the API raises weird errors, e.g. input=[[126]]
                    if decode:
                        tokens = self.tokenizer.decode(tokens)
                    all_tokens.append(tokens)
                    used_indices.append(j)

                out = [[]] * len(batch)
                if all_tokens:
                    response = openai.Embedding.create(input=all_tokens, model=self.engine)
                    # May want to sleep here to avoid getting too many requests error
                    # time.sleep(1)
                    assert len(response["data"]) == len(
                        all_tokens
                    ), f"Sent {len(all_tokens)}, got {len(response['data'])}"

                    for data in response["data"]:
                        idx = data["index"]
                        # OpenAI seems to return them ordered, but to be save use the index and insert
                        idx = used_indices[idx]
                        embedding = data["embedding"]
                        out[idx] = embedding

                fin_embeddings.extend(out)
        # Save embeddings
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--engine", type=str, default="text-embedding-ada-002")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    args = parser.parse_args()
    return args

def main(args):

    # There are two different batch sizes
    # OpenAIEmbedder(...) batch size arg is used to send X embeddings to the API
    # evaluation.run(...) batch size arg is how much will be saved / pickle file (as it's the total sent to the embed function)


    model = OpenAIEmbedder(args.engine, task_name=task, batch_size=args.batchsize, save_emb=True)
    eval_splits = ['test']
    model_name = args.engine.split("/")[-1].split("_")[-1]
    evaluation = MTEB(task_langs=["es"])
    evaluation.run(model, output_folder=f"results/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits, corpus_chunk_size=10000)

if __name__ == "__main__":
    args = parse_args()
    main(args)