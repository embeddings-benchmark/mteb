from collections import defaultdict
from typing import Dict, Tuple
import os
import logging
from datasets import load_dataset, Value, Features
from tqdm.autonotebook import tqdm
import json
import csv

logger = logging.getLogger(__name__)


class HFDataLoader:
    
    def __init__(self, hf_repo: str = None, hf_repo_qrels: str = None, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = "", streaming: bool = False, keep_in_memory: bool = False):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.hf_repo = hf_repo
        if hf_repo:
            logger.warn("A huggingface repository is provided. This will override the data_folder, prefix and *_file arguments.")
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo + "-qrels"
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

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        if not self.hf_repo:
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        self._load_qrels(split)
        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row['query-id']][row['corpus-id']] = int(row['score'])
        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x['id'] in self.qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])
        
        return self.corpus, self.queries, self.qrels
    
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
            print(corpus_ds)
            
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
            print(queries_ds)
            
        else:
            queries_ds = load_dataset('json', data_files=self.query_file, streaming=self.streaming, keep_in_memory=self.keep_in_memory)
        queries_ds = next(iter(queries_ds.values())) # get first split
        queries_ds = queries_ds.cast_column('_id', Value('string'))
        queries_ds = queries_ds.rename_column('_id', 'id')
        queries_ds = queries_ds.remove_columns([col for col in queries_ds.column_names if col not in ['id', 'text']])
        self.queries = queries_ds
        
    def _load_qrels(self, split):
        if self.hf_repo:
            qrels_ds = load_dataset(self.hf_repo_qrels, keep_in_memory=self.keep_in_memory, streaming=self.streaming)[split]
            print(qrels_ds)
            
        else:
            qrels_ds = load_dataset('csv', data_files=self.qrels_file, delimiter='\t', keep_in_memory=self.keep_in_memory)
        features = Features({'query-id': Value('string'), 'corpus-id': Value('string'), 'score': Value('float')})
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds




logger = logging.getLogger(__name__)

class GenericDataLoader:
    
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else qrels_folder
        self.qrels_file = qrels_file
    
    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus
    
    def _load_corpus(self):
    
        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }
    
    def _load_queries(self):
        
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score