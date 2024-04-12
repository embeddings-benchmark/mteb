import logging
import json
import os
from collections import defaultdict
from time import time
from typing import Dict, Tuple
import tqdm

from datasets import load_dataset, Value, Features

from ..evaluation.evaluators import InstructionRetrievalEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)

# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
class HFDataLoader:    
    def __init__(self, hf_repo: str = None, hf_repo_qrels: str = None, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = "", streaming: bool = False, keep_in_memory: bool = False):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.og_instructions = {}
        self.changed_instructions = {}
        self.top_ranked = {}
        self.hf_repo = hf_repo
        if hf_repo:
            # By default fetch qrels from same repo not a second repo with "-qrels" like in original
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo
        else:
            # data folder would contain these files: 
            # (1) fiqa/corpus.jsonl  (format: jsonlines)
            # (2) fiqa/queries.jsonl (format: jsonlines)
            # (3) fiqa/qrels/test.tsv (format: tsv ("\t"))
            if prefix:
                query_file = prefix + "-" + query_file
                qrels_folder = prefix + "-" + qrels_folder

            self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
            self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
            self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
            self.qrels_file = qrels_file
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory
    
    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        if not self.hf_repo:
            self.og_qrels_file = os.path.join(self.qrels_folder + "_og", split + ".tsv")
            self.changed_qrels_file = os.path.join(self.qrels_folder + "_changed", split + ".tsv")
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.og_qrels_file, ext="tsv")
            self.check(fIn=self.changed_qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        self._load_qrels(split, changed=False)
        self._load_qrels(split, changed=True)
        # filter queries with no qrels
        og_qrels_dict = defaultdict(dict)
        changed_qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            og_qrels_dict[row['query-id']][row['corpus-id']] = int(row['score'])

        def qrels_changed_dict_init(row):
            changed_qrels_dict[row['query-id']][row['corpus-id']] = int(row['score'])
        
        self.changed_qrels.map(qrels_dict_init)
        self.og_qrels.map(qrels_changed_dict_init)
        self.og_qrels = og_qrels_dict
        self.changed_qrels = changed_qrels_dict
        self.queries = self.queries.filter(lambda x: x['id'] in self.og_qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])

        # load top_ranked
        self.load_top_ranked()
        
        return self.corpus, self.queries, self.og_qrels, self.changed_qrels, self.top_ranked
    

    def load_top_ranked(self) -> Dict[str, Dict[str, str]]:
        if self.hf_repo:
            top_ranked_ds = load_dataset(self.hf_repo, 'top_ranked', keep_in_memory=self.keep_in_memory, streaming=self.streaming)
        else:
            top_ranked_ds = load_dataset('json', data_files=self.top_ranked_file, streaming=self.streaming, keep_in_memory=self.keep_in_memory)
        top_ranked_ds = next(iter(top_ranked_ds.values())) # get first split
        top_ranked_ds = top_ranked_ds.cast_column('qid', Value('string'))
        top_ranked_ds = top_ranked_ds.cast_column('pid', Value('string'))
        top_ranked_ds = top_ranked_ds.remove_columns([col for col in top_ranked_ds.column_names if col not in ['qid', 'pid']])
        self.top_ranked = top_ranked_ds

    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        if not self.hf_repo:
            self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus
    
    def _load_corpus(self):
        if self.hf_repo:
            corpus_ds = load_dataset(self.hf_repo, 'corpus', keep_in_memory=self.keep_in_memory, streaming=self.streaming)
        else:
            corpus_ds = load_dataset('json', data_files=self.corpus_file, streaming=self.streaming, keep_in_memory=self.keep_in_memory)
        corpus_ds = next(iter(corpus_ds.values())) # get first split
        corpus_ds = corpus_ds.cast_column('_id', Value('string'))
        corpus_ds = corpus_ds.rename_column('_id', 'id')
        corpus_ds = corpus_ds.remove_columns([col for col in corpus_ds.column_names if col not in ['id', 'text', 'title']])
        self.corpus = corpus_ds
    
    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(self.hf_repo, 'queries', keep_in_memory=self.keep_in_memory, streaming=self.streaming)
        else:
            queries_ds = load_dataset('json', data_files=self.query_file, streaming=self.streaming, keep_in_memory=self.keep_in_memory)
        queries_ds = next(iter(queries_ds.values())) # get first split
        queries_ds = queries_ds.cast_column('_id', Value('string'))
        queries_ds = queries_ds.rename_column('_id', 'id')
        queries_ds = queries_ds.remove_columns([col for col in queries_ds.column_names if col not in ['id', 'text', 'instruction_og', 'instruction_changed', "keywords", "short_query"]])
        self.queries = queries_ds
        
    def _load_qrels(self, split, changed=False):
        if self.hf_repo:
            qrels_ds = load_dataset(self.hf_repo_qrels, "qrels_og" if not changed else "qrels_changed", keep_in_memory=self.keep_in_memory, streaming=self.streaming)[split]
        else:
            qrels_file = self.og_qrels_file if not changed else self.changed_qrels_file
            qrels_ds = load_dataset('csv', data_files=qrels_file, delimiter='\t', keep_in_memory=self.keep_in_memory)
        features = Features({'query-id': Value('string'), 'corpus-id': Value('string'), 'score': Value('float')})
        qrels_ds = qrels_ds.cast(features)

        if changed:
            self.changed_qrels = qrels_ds
        else:
            self.og_qrels = qrels_ds


class AbsTaskInstructionRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_length_ablation = kwargs.get("do_length_ablation", False)

    def load_data(self, **kwargs):
        if self.data_loaded: return
        self.corpus, self.queries, self.og_relevant_docs, self.changed_relevant_docs = {}, {}, {}, {}
        self.og_instructions, self.changed_instructions = {}, {}
        self.top_ranked = {}
        self.keywords, self.short_instructions = {}, {}
        hf_repo_qrels = self.description["hf_hub_name"] + "-qrels" if "clarin-knext" in self.description["hf_hub_name"] else None
        for split in kwargs.get("eval_splits", self.description["eval_splits"]):
            corpus, queries, og_qrels, changed_qrels, top_ranked_init = HFDataLoader(hf_repo=self.description["hf_hub_name"], hf_repo_qrels=hf_repo_qrels, streaming=False, keep_in_memory=False).load(split=split)
            # Conversion from DataSet
            top_ranked = defaultdict(list)
            [top_ranked[cur_inst["qid"]].append(cur_inst["pid"]) for cur_inst in top_ranked_init]
            og_instructions = {query["text"]: query["instruction_og"] for query in queries}
            changed_instructions = {query["text"]: query["instruction_changed"] for query in queries}
            keywords = {query['text']: query['keywords'] for query in queries}
            short_instructions = {query['text']: query['short_query'] for query in queries}
            queries = {query['id']: query['text'] for query in queries}
            assert len(top_ranked) == len(queries), f"Top ranked not loaded properly! Expected {len(self.queries)} but got {len(self.top_ranked)}."
            corpus = {doc['id']: {'title': doc['title'] , 'text': doc['text']} for doc in corpus}
            self.corpus[split], self.queries[split], self.og_relevant_docs[split], self.changed_relevant_docs[split] = corpus, queries, og_qrels, changed_qrels
            self.changed_instructions[split], self.og_instructions[split] = changed_instructions, og_instructions
            self.top_ranked[split] = top_ranked
            self.keywords[split], self.short_instructions[split] = keywords, short_instructions

        self.data_loaded = True

    def evaluate(
        self,
        model,
        split="test",
        **kwargs
    ):
        retriever = InstructionRetrievalEvaluator(model, **kwargs)

        scores_og = {}
        scores_changed = {}
        if self.is_multilingual:
            for lang in self.langs:
                logger.info(f"Language: {lang}")
                corpus, queries, og_relevant_docs, changed_relevant_docs = self.corpus[lang][split], self.queries[lang][split], self.og_relevant_docs[lang][split], self.changed_relevant_docs[lang][split]
                og_instructions, changed_instructions = self.og_instructions[lang][split], self.changed_instructions[lang][split]
                keywords, short_instructions = self.keywords[lang][split], self.short_instructions[lang][split]
                top_ranked = self.top_ranked[lang][split]
                scores_og[lang], results_og[lang] = self._evaluate_monolingual(retriever, corpus, queries, og_relevant_docs, "og", og_instructions, top_ranked, lang, **kwargs)
                scores_changed[lang], results_changed[lang] = self._evaluate_monolingual(retriever, corpus, queries, changed_relevant_docs, "changed", changed_instructions, top_ranked, lang, **kwargs)

                scores_base[lang], results_base[lang] = self._evaluate_monolingual(retriever, corpus, queries, og_relevant_docs, "base", defaultdict(str), top_ranked, lang, **kwargs)

                newly_irrelevant_qrels = self.create_qrel_diff(self.og_relevant_docs[lang][split], self.changed_relevant_docs[lang][split])
                changed_scores[lang] = retriever.evaluate_change(results_og[lang], results_changed[lang], newly_irrelevant_qrels)

                changed_scores[lang]["individual"] = {
                    "original": scores_og[lang],
                    "changed": scores_changed[lang],
                    "base": scores_base[lang]
                }
        else:
            corpus, queries, og_relevant_docs, changed_relevant_docs = self.corpus[split], self.queries[split], self.og_relevant_docs[split], self.changed_relevant_docs[split]
            og_instructions, changed_instructions = self.og_instructions[split], self.changed_instructions[split]
            top_ranked = self.top_ranked[split]
            keywords, short_instructions = self.keywords[split], self.short_instructions[split]
            scores_og, results_og = self._evaluate_monolingual(retriever, corpus, queries, og_relevant_docs, "og", og_instructions, top_ranked, None, **kwargs)
            scores_changed, results_changed = self._evaluate_monolingual(retriever, corpus, queries, changed_relevant_docs, "changed", changed_instructions, top_ranked, None, **kwargs)
            scores_base, results_base = self._evaluate_monolingual(retriever, corpus, queries, og_relevant_docs, "base", defaultdict(str), top_ranked, None, **kwargs)

            newly_irrelevant_qrels = self.create_qrel_diff(self.og_relevant_docs[split], self.changed_relevant_docs[split])
            changed_scores = retriever.evaluate_change(results_og, results_changed, newly_irrelevant_qrels)            

            changed_scores["individual"] = {
                "original": scores_og,
                "changed": scores_changed,
                "base": scores_base
            }

            if self.do_length_ablation:
                scores_w_keywords = self._evaluate_monolingual(retriever, corpus, queries, og_relevant_docs, "keywords", keywords, top_ranked, None, **kwargs)
                scores_w_short_instr = self._evaluate_monolingual(retriever, corpus, queries, og_relevant_docs, "short_instructions", short_instructions, top_ranked, None, **kwargs)
                changed_scores["length_ablation"] = {
                    "keywords": scores_w_keywords,
                    "short_instructions": scores_w_short_instr
                }

        return changed_scores

    def _evaluate_monolingual(self, retriever, corpus, queries, relevant_docs, qrels_name: str, instructions, top_ranked, lang=None, **kwargs):
        start_time = time()

        # do the results by query and relevant docs only
        all_results = []
        for query_id in tqdm.tqdm(list(queries.keys()), leave=True):
            cur_queries = {query_id: queries[query_id]}
            cur_instructions = {queries[query_id]: instructions[queries[query_id]]} 
            cur_docs = {key: value for (key, value) in corpus.items() if key in top_ranked[query_id]}
            all_results.append(retriever(cur_docs, cur_queries, instructions=cur_instructions, qid=query_id))

        # combine all the results (which are {'qid' -> {'doc_id' -> score} mappings)
        # we know all are unique qids, so we can smash together
        results = {k: v for d in all_results for k, v in d.items()}

        end_time = time()
        logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        if kwargs.get("save_qrels", False):
            output_folder = kwargs.get("output_folder", "results")
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            top_k = kwargs.get('top_k', None)
            if top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(sorted(results[qid], key=lambda x: results[qid][x], reverse=True)[:top_k])
                    results[qid] = {k: v for k, v in results[qid].items() if k in doc_ids}
            if lang is None:
                qrels_save_path = f"{output_folder}/{self.description['name']}_qrels_{qrels_name}.json"
            else:
                qrels_save_path = f"{output_folder}/{self.description['name']}_{lang}_qrels_{qrels_name}.json"
            
            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values, ignore_identical_ids=kwargs.get("ignore_identical_ids", True))
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores, results


    def create_qrel_diff(self, og_qrels, changed_qrels):
        newly_irrelevant_qrels = {}
        for qid in og_qrels:
            newly_irrelevant_qrels[qid] = []
            for doc_id in og_qrels[qid]:
                if changed_qrels[qid][doc_id] != og_qrels[qid][doc_id]:
                    newly_irrelevant_qrels[qid].append(doc_id)

        return newly_irrelevant_qrels
