import argparse
import copy
import os
import random
import sqlparse
import torch
import json
import sqlite3
import numpy as np
import transformers
import pandas as pd
from tqdm import trange
from bridge_content_encoder import get_database_matches

LABEL_MAP = {
    'mnli': {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    },
    'rte': {
        0: 'entailment',
        1: 'contradiction'
    },
    'mrpc': {
        1: 'equivalent',
        0: 'not equivalent'
    },
    'sst5': {
        0: 'very negative',
        1: 'negative',
        2: 'neutral',
        3: 'positive',
        4: 'very positive'
    },
    'dbpedia_14': {
        0: 'company',
        1: 'education',
        2: 'artist',
        3: 'athlete',
        4: 'officeHolder',
        5: 'transportation',
        6: 'building',
        7: 'nature',
        8: 'village',
        9: 'animal',
        10: 'plant',
        11: 'album',
        12: 'film',
        13: 'book',
    },
}
INSTRUCTOR_PROMPTS = {
    'mnli': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'rte': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'mrpc': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'sst5': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'dbpedia_14': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'nq': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'xsum': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'hellaswag': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
    'geoquery': {
        'query': 'Represent question:',
        'doc': 'Represent question:'
    },
    'mwoz': {
        'query': 'Represent text:',
        'doc': 'Represent text:'
    },
}
GRITLM_PROMPTS = {
    'mnli': {
        'query': 'Given a sentence, retrieve a similar sentence',
    },
    'rte': {
        'query': 'Given a sentence, retrieve a similar sentence',
    },
    'mrpc': {
        'query': 'Given two sentences, retrieve two similar sentences',
    },
    'sst5': {
        'query': 'Given a sentence, retrieve a similar sentence',
    },
    'dbpedia_14': {
        'query': 'Given a sentence, retrieve a similar sentence',
    },
    'nq': {
        'query': 'Given a question, retrieve a similar question',
    },
    'xsum': {
        'query': 'Given a paragraph, retrieve a similar paragraph',
    },
    'hellaswag': {
        'query': 'Given a sentence, retrieve a similar sentence',
    },
    'geoquery': {
        'query': 'Given a question, retrieve a similar question',
    },
    'mwoz': {
        'query': 'Given a conversation, retrieve a similar conversation',
    }
}
SYSTEM_MESSAGES = {
    'mnli': "You are a helpful assistant good at determining whether a claim is true.",
    'rte': "You are a helpful assistant good at determining relationships between sentences.",
    'mrpc': "You are a helpful assistant good at determining whether two sentences are equivalent.",
    'sst5': "You are a helpful assistant good at determining the sentence emotion.",
    'dbpedia_14': "You are a helpful assistant good at classifying sentence topic.",
    'nq': "You are a helpful assistant good at answering questions.",
    'xsum': "You are a helpful assistant good at summarization.",
    'hellaswag': "You are a helpful assistant good at finding the sentence ending.",
    'geoquery': "You are a helpful assistant good at writing SQL."
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def maybe_add_quotes(val):
    if isinstance(val, str):
        return "'" + val + "'"
    return str(val)

def get_db_schemas():
    with sqlite3.connect(f'data/geoquery.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            schemas[table[0]] = cursor.fetchone()[0]
        return schemas

def get_db_rows(*, rows=3, db_content_matching=True, question=None):
    db_path = f'data/geoquery.sqlite'
    results = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            cursor.execute("PRAGMA table_info({})".format(table[0]))
            results[table[0]] = pd.read_sql_query(f"SELECT * FROM {table[0]} LIMIT {rows}", conn)
        if db_content_matching:
            for table in results.keys():
                where_clauses = list()
                for col in results[table].keys():
                    matches = get_database_matches(question, table, col, db_path)
                    for match in matches:
                        where_clause = f'{col} = {maybe_add_quotes(match)}'
                        where_clauses.append(where_clause)
                if len(where_clauses) > 0:
                    table_matches = pd.read_sql_query(
                        f"SELECT DISTINCT * FROM {table} WHERE {' OR '.join(where_clauses)} LIMIT {rows}", conn)
                    results[table] = table_matches
    for k, v in results.items():
        results[k] = v.to_string(index=False)
    return results

def get_db_prompt(*, schema=True, rows=0, db_content_matching=True,question=None, reindent_aligned=True):
    schemas = get_db_schemas()
    examples = get_db_rows(rows=rows, db_content_matching=db_content_matching, question=question)
    prompt = ''
    if schema or (rows > 0):
        for table in schemas.keys():
            if schema:
                prompt += sqlparse.format(schemas[table], reindent_aligned=reindent_aligned)
                prompt += '\n'
            if rows > 0:
                prompt += '/*\n'
                # prompt += f'{rows} example rows from table {table}:\n'
                # prompt += f'SELECT * FROM {table} LIMIT {rows};\n'
                if not schema:
                    prompt += f'Table: {table}\n'
                prompt += examples[table]
                prompt += '\n*/\n'
            prompt += '\n'
    return prompt

def get_prompt_instructions():
    return "-- Using valid SQLite, answer the following questions for the tables provided above.\n"

def format_example_mnli(e):
    return {
        'with_label': f"Premise: {e['premise']}\n\n"
                     f"Hypothesis: {e['hypothesis']}\n\n"
                     f"Based on that information:\n"
                     f"Answer with entailment if the hypothesis is entailment;\n"
                     f"Answer with contradiction if the hypothesis is contradiction;\n"
                     f"Answer with neutral if the hypothesis is neutral;\n"
                     f"The answer is: {LABEL_MAP['mnli'][e['label']]}",
        'without_label': f"Premise: {e['premise']}\n\n"
                      f"Hypothesis: {e['hypothesis']}\n\n"
                      f"Based on that information:\n"
                      f"Answer with entailment if the hypothesis is entailment;\n"
                      f"Answer with contradiction if the hypothesis is contradiction;\n"
                      f"Answer with neutral if the hypothesis is neutral;\n"
                      f"The answer is: ",
        'raw': f"Premise: {e['premise']}\n\n"
                f"Hypothesis: {e['hypothesis']}\n\n"
    }

def format_example_rte(e):
    return {
        'with_label': f"Sentence 1: {e['sentence1']}\n\n"
                     f"Sentence 2: {e['sentence2']}\n\n"
                     f"Answer with entailment if sentence 1 entails sentence 2;\n"
                     f"Answer with contradiction if sentence 1 contradicts sentence 2;\n"
                     f"The answer is: {LABEL_MAP['rte'][e['label']]}",
        'without_label': f"Sentence 1: {e['sentence1']}\n\n"
                     f"Sentence 2: {e['sentence2']}\n\n"
                     f"Answer with entailment if sentence 1 entails sentence 2;\n"
                     f"Answer with contradiction if sentence 1 contradicts sentence 2;\n"
                     f"The answer is: ",
        'raw': f"Sentence 1: {e['sentence1']}\n\n"
                f"Sentence 2: {e['sentence1']}\n\n"
    }

def format_example_hellaswag(e):
    m = ['(A)','(B)','(C)','(D)']
    return {
        'with_label':f"Sentence: The topic is {e['activity_label']}. {e['ctx_a']} {e['ctx_b']} ...\n"
                     f"(A). {e['endings'][0]}\n"
                      f"(B). {e['endings'][1]}\n"
                      f"(C). {e['endings'][2]}\n"
                      f"(D). {e['endings'][3]}\n"
                     f"The ending is {m[int(e['label'])]}",
        'without_label': f"Sentence: The topic is {e['activity_label']}. {e['ctx_a']} {e['ctx_b']} ...\n"
                      f"Please choose one of the following as the ending of the last sentence:\n"
                     f"(A). {e['endings'][0]}\n"
                      f"(B). {e['endings'][1]}\n"
                      f"(C). {e['endings'][2]}\n"
                      f"(D). {e['endings'][3]}",
        'raw': f"Sentence: The topic is {e['activity_label']}. {e['ctx_a']} {e['ctx_b']} ...\n"
             f"(A). {e['endings'][0]}\n"
              f"(B). {e['endings'][1]}\n"
              f"(C). {e['endings'][2]}\n"
              f"(D). {e['endings'][3]}"
    }

def format_example_mrpc(e):
    return {
        'with_label': f"Sentence 1: {e['sentence1']}\n\n"
                     f"Sentence 2: {e['sentence2']}\n\n"
                     f"Answer with equivalent if sentence 1 and sentence 2 are equivalent;\n"
                     f"Answer with not equivalent if sentence 1 and sentence 2 are not equivalent;\n"
                     f"The answer is: {LABEL_MAP['mrpc'][e['label']]}",
        'without_label': f"Sentence 1: {e['sentence1']}\n\n"
                     f"Sentence 2: {e['sentence2']}\n\n"
                     f"Answer with equivalent if sentence 1 and sentence 2 are equivalent;\n"
                     f"Answer with not equivalent if sentence 1 and sentence 2 are not equivalent;\n"
                     f"The answer is: ",
        'raw': f"Sentence 1: {e['sentence1']}\n\n"
                f"Sentence 2: {e['sentence1']}\n\n"
    }

def format_example_sst5(e):
    return {
        'with_label': f"Sentence: {e['text']}\n"
                      f"The emotion is {LABEL_MAP['sst5'][e['label']]}\n\n\n",
        'without_label': f"Sentence: {e['text']}\n"
                         f"Is the sentence \"very negative\", \"negative\", \"neutral\", \"positive\" or \"very positive\"?\n"
                         f"The emotion is ",
        'raw': e['text']
    }

def format_example_nq(e):
    return {
        'with_label': f"Question: {e['question']}\n"
                      f"Answer: {e['answers'][0]}\n\n\n",
        'without_label': f"Question: {e['question']}\n"
                      f"Answer: ",
        'raw': e['question']
    }

def format_example_dbpedia_14(e):
    return {
        'with_label': f"Sentence: {e['title']} {e['content']}\n"
                      f"The topic is {LABEL_MAP['dbpedia_14'][e['label']]}\n\n\n",
        'without_label': f"Sentence: {e['title']} {e['content']}\n"
                         f"Which topic does the sentence belong to? company, education, artist, athlete, officeHolder, transportation, building, nature, village, animal, plant, album, film or book?\n"
                         f"The topic is ",
        'raw': e['title']+' '+e['content']
    }

def format_example_xsum(e):
    return {
        'with_label': f"Document: {e['document']}\n"
                      f"Summarization: {e['summary']}\n\n\n",
        'without_label': f"Document: {e['document']}\n"
                         f"Please write the summary of the last document.\n"
                         f"Summarization: ",
        'raw': e['document']
    }

def format_example_geoquery(e):
    return {
        'with_label': f"Question: {e['question']}\n"
                      f"SQL: {e['query']}\n\n\n",
        'without_label': f"Question: {e['question']}\n"
                      f"Using valid SQLite, write the SQL for the last question.\n"
                         f"Wrap the SQL with ```\n",
        'raw': e['question']
    }

def retrieval_sentence_bert(model_id,queries,docs,batch_size=128,top_k=24,**kwargs):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    model = SentenceTransformer(model_id)
    query_emb = model.encode(queries,batch_size=batch_size)
    doc_emb = model.encode(docs,batch_size=batch_size)
    scores = cosine_similarity(query_emb, doc_emb)
    retrieved_ids = []
    for s in scores:
        retrieved_ids.append(np.argsort(s).tolist()[-top_k:])
    return retrieved_ids

def retrieval_instructor(model_id,queries,docs,batch_size=128,top_k=24,**kwargs):
    from sklearn.metrics.pairwise import cosine_similarity
    from InstructorEmbedding import INSTRUCTOR
    model = INSTRUCTOR(model_id)
    queries = [[INSTRUCTOR_PROMPTS[kwargs['task']]['query'],q] for q in queries]
    docs = [[INSTRUCTOR_PROMPTS[kwargs['task']]['doc'],d] for d in docs]
    query_emb = model.encode(queries,batch_size=batch_size)
    doc_emb = model.encode(docs,batch_size=batch_size)
    scores = cosine_similarity(query_emb, doc_emb)
    retrieved_ids = []
    for s in scores:
        retrieved_ids.append(np.argsort(s).tolist()[-top_k:])
    return retrieved_ids

def retrieval_bm25(model_id,queries,docs,batch_size=128,top_k=24,**kwargs):
    from rank_bm25 import BM25Okapi
    corpus = docs
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25_model = BM25Okapi(tokenized_corpus)
    retrieved_ids = []
    for q in queries:
        tokenized_query = q.split(" ")
        doc_scores = bm25_model.get_scores(tokenized_query)
        retrieved_ids.append(np.argsort(doc_scores).tolist()[-top_k:])
    return retrieved_ids

def retrieval_gritlm(model_id,queries,docs,batch_size=128,top_k=24,**kwargs):
    from gritlm import GritLM
    from sklearn.metrics.pairwise import cosine_similarity
    model = GritLM(model_id, torch_dtype="auto", mode="embedding")
    def gritlm_instruction(instruction):
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    # No need to add instruction for retrieval documents
    doc_emb = model.encode(docs, instruction=gritlm_instruction(""), batch_size=batch_size)
    query_emb = model.encode(queries, instruction=gritlm_instruction(GRITLM_PROMPTS[kwargs['task']]['query']), batch_size=batch_size)
    scores = cosine_similarity(query_emb, doc_emb)
    print(scores.shape)
    retrieved_ids = []
    for s in scores:
        retrieved_ids.append(np.argsort(s).tolist()[-top_k:])
    return retrieved_ids

RETRIEVAL_FUNCS = {
    'sentence-transformers/all-mpnet-base-v2': retrieval_sentence_bert,
    'hkunlp/instructor-large': retrieval_instructor,
    'hkunlp/instructor-base': retrieval_instructor,
    'hkunlp/instructor-xl': retrieval_instructor,
    'bm25': retrieval_bm25,
    'GritLM/GritLM-7B': retrieval_gritlm
}
FORMAT_FUNCS = {
    'mnli': format_example_mnli,
    'rte': format_example_rte,
    'mrpc': format_example_mrpc,
    'sst5': format_example_sst5,
    'dbpedia_14': format_example_dbpedia_14,
    'nq': format_example_nq,
    'xsum': format_example_xsum,
    'hellaswag': format_example_hellaswag,
    'geoquery': format_example_geoquery
}

if __name__=='__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True,type=str,choices=['mnli','rte','mrpc','sst5','dbpedia_14','nq','xsum','hellaswag','geoquery'])
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--batch_size', type=int,default=16)
    parser.add_argument('--emb_batch_size', type=int, default=16)
    parser.add_argument('--retrieval', type=str,default=None)
    args = parser.parse_args()
    
    args.output_dir = os.path.join(args.output_dir,args.task)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    with open(f'data/{args.task}_train.json') as f:
        train_examples = json.load(f)
    with open(f'data/{args.task}_eval.json') as f:
        eval_examples = json.load(f)
    if args.retrieval is not None:
        queries = [FORMAT_FUNCS[args.task](e)['raw'] for e in eval_examples]
        docs = [FORMAT_FUNCS[args.task](e)['raw'] for e in train_examples]
        if args.task in ['sst5','nq','xsum','hellaswag']:
            topk = 4
        elif args.task in ['xsum']:
            topk = 1
        else:
            topk = 24
        retrieved_ids = RETRIEVAL_FUNCS[args.retrieval](model_id=args.retrieval,queries=queries,docs=docs,task=args.task,
                                                        batch_size=args.emb_batch_size,top_k=topk)
        contexts = []
        for example_id,ids in enumerate(retrieved_ids):
            cur_context = ''
            for idx in ids:
                formatted_example = FORMAT_FUNCS[args.task](train_examples[idx])['with_label'].strip()+'\n\n'
                cur_context += formatted_example
            contexts.append(cur_context)
    else:
        contexts = ['' for _ in range(len(eval_examples))]
    assert len(contexts)==len(eval_examples),f"{len(contexts)},{len(eval_examples)}"
    
    if args.task in ['geoquery']:
        model_id = "Qwen/CodeQwen1.5-7B-Chat"
    else:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print('model:',model_id)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pad_token_id = pipeline.tokenizer.eos_token_id
    
    inference_data = [[FORMAT_FUNCS[args.task](e)['without_label'],e['idx']] for e in eval_examples]
    global_count = 0
    for i in trange(0,len(inference_data),args.batch_size):
        cur_batch = []
        cur_indices = []
        for example in inference_data[i:i+args.batch_size]:
            assert example[1]==eval_examples[example[1]]['idx'],f"{example[1]},{eval_examples[example[1]]['idx']}"
            if args.task=='geoquery':
                db_content = get_db_prompt(rows=3,question=eval_examples[example[1]]['question'])
                contexts[example[1]] = db_content + contexts[example[1]]
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGES[args.task]},
                {"role": "user", "content": contexts[example[1]]+example[0]},
            ]
            formatted_prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompt_ids = pipeline.tokenizer(formatted_prompt)['input_ids']
            if len(formatted_prompt_ids)>8000:
                formatted_prompt = pipeline.tokenizer.decode(formatted_prompt_ids[-8000:])
            cur_batch.append(formatted_prompt)
            cur_indices.append(example[1])
        # print(cur_batch[0])
        if args.task in ['xsum','geoquery']:
            max_new_tokens = 256
        else:
            max_new_tokens = 30
        batch_outputs = pipeline(
            cur_batch,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=pad_token_id,
            temperature=None,
            top_p=None
        )
        assert len(batch_outputs)==len(cur_indices),f"{len(batch_outputs)}, {len(cur_indices)}"
        for idx,cur_input,cur_output in zip(cur_indices,cur_batch,batch_outputs):
            example = copy.deepcopy(eval_examples[idx])
            example['pred'] = cur_output[0]["generated_text"][len(cur_input):]
            # if global_count<1:
            #     print('input')
            #     print(cur_input)
            #     print('output')
            #     print(example['pred'])
            global_count += 1
            assert idx==eval_examples[idx]['idx'],f"{idx},{eval_examples[idx]['idx']}"
            with open(os.path.join(args.output_dir,f"{idx}.json"),'w') as f:
                json.dump(example,f,indent=2)
    
    
    
    
    
