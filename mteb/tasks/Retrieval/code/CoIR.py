from __future__ import annotations
import json, csv
from io import StringIO
from datasets import load_dataset, Value, Features
from mteb.abstasks import AbsTaskRetrieval, TaskMetadata, HFDataLoader

_EVAL_SPLIT = "test"

class CoIRTasks:
    def _load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        _corpus, _queries = {}, {}
        task_name = self.metadata_dict["dataset"]["name"]
        queries_corpus_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-queries-corpus")
        qrels_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-qrels")

        corpus_data = [q for q in queries_corpus_dataset['corpus']]
        query_data = [q for q in queries_corpus_dataset['queries'] if q['partition'] == _EVAL_SPLIT]
        
        # corpus handling 
        corpus_file = StringIO('\n'.join(json.dumps(doc) for doc in corpus_data))
        corpus_file.seek(0)
        for line in corpus_file:
            doc = json.loads(line)
            _corpus[doc["_id"]] = {
                "text": doc.get("text"),
                "title": doc.get("title")
            }
            
        query_file = StringIO('\n'.join(json.dumps(doc) for doc in query_data))
        query_file.seek(0)
        for line in query_file:
            doc = json.loads(line)
            _queries[doc["_id"]] = {
                "text": doc.get("text"),
                "title": doc.get("title")
            }

        # qrels_data = qrels_dataset['test']
        
        for split in kwargs.get("eval_splits", self.metadata_dict["eval_splits"]):
            qrel_split = qrels_dataset[split]
            qrels_file = StringIO('\n'.join(f"{qrel['query_id']}\t{qrel['corpus_id']}\t{qrel['score']}" for qrel in qrel_split))
            qrels_file.seek(0)
            reader = csv.reader(qrels_file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            qrels = {}
            for row in reader:
                query_id, corpus_id, score = row[0], row[1], int(row[2])
                if query_id not in qrels:
                    qrels[query_id] = {corpus_id: score}
                else:
                    qrels[query_id][corpus_id] = score
            self.relevant_docs[split] = qrels
            self.queries[split] = _queries
            self.corpus[split] = _corpus

        self.data_loaded = True
            
class CodeContestRetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeContestRetrieval",
        dataset={
            "path": "CoIR-Retrieval/apps",
            "revision": "",
            "name": "apps",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving relevant code solutions in response to coding problems described in natural language"),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-11-08"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 3765}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()
        
class SyntheticText2SQLRetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SyntheticText2SQL",
        dataset={
            "path": "CoIR-Retrieval/synthetic-text2sql",
            "revision": "",
            "name": "synthetic-text2sql",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving relevant SQL code in response to natural language questions"),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-11-08"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 5851}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()
        
class CodeTranslationContestRetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeTranslationContestRetrieval",
        dataset={
            "path": "CoIR-Retrieval/codetrans-contest",
            "revision": "",
            "name": "codetrans-contest",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving semantically equivalent C++ code from its counterpart written in Python."),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-11-08"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 221}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()

class CodeTranslationDLRetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeTranslationDLRetrieval",
        dataset={
            "path": "CoIR-Retrieval/codetrans-dl",
            "revision": "",
            "name": "codetrans-dl",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving semantically equivalent code written using 2 different machine learning frameworks"),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-11-08"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 180}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()
        
class StackoverflowQARetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StackoverflowQARetrieval",
        dataset={
            "path": "CoIR-Retrieval/stackoverflow-qa",
            "revision": "",
            "name": "stackoverflow-qa",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving relevant answer in text with some code snippets given a natural language question"),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2024-05-01"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 221}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()
        
class CodeFeedbackMTRetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeFeedbackMTRetrieval",
        dataset={
            "path": "CoIR-Retrieval/codefeedback-mt",
            "revision": "",
            "name": "codefeedback-mt",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving relevant answer in text with some code snippets given a natural language question and preceding dialogue history."),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-11-08"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 13277}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()
        
class CodeFeedbackSTRetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeFeedbackSTRetrieval",
        dataset={
            "path": "CoIR-Retrieval/codefeedback-st",
            "revision": "",
            "name": "codefeedback-mt",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving relevant answer in text with some code snippets given a natural language question."),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-11-08"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 31306}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()
        
class WebQueryCodeRetrieval(CoIRTasks, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WebQueryCodeRetrieval",
        dataset={
            "path": "CoIR-Retrieval/cosqa",
            "revision": "",
            "name": "cosqa",
        },
        reference="https://huggingface.co/CoIR-Retrieval",
        description=("Retrieving relevant code solutions in response to natural language problems."),
        type="Retrieval",
        modalities=['text'],
        sample_creation="found",
        category="p2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["python-Code"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-11-08"),  
        form=["written"],
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="Not specified",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
            @misc{li2024coircomprehensivebenchmarkcode,
            title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models}, 
            author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
            year={2024},
            eprint={2407.02883},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2407.02883}, 
        }
        """,
        descriptive_stats={
            "n_samples":{_EVAL_SPLIT: 31306}   
        }
    )
    
    def load_data(self, **kwargs):
        self._load_data()