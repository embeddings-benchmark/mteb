import argparse
from typing import List, Dict, Any, Tuple
import pickle as pkl
import tqdm
from exec_eval import exec_on_db, result_eq
import os
from collections import defaultdict
import time
from multiprocessing import cpu_count, Pool, Manager
from itertools import repeat

NUM_PROCESSES = cpu_count() // 3
if NUM_PROCESSES == 0:
    NUM_PROCESSES = 1
MULTIPLICATIVE_OVERHEAD = 3
ADDITIVE_OVERHEAD = 30
GOLD_TIMEOUT = 100

cache_path = 'cache.pkl'
m = Manager()
cache = m.dict()

def load_predictions(f_path: str) -> List[str]:
    preds = []
    with open(f_path, 'r') as in_file:
        for l in in_file:
            preds.append(l.strip())
    return preds


def acc(l, idxes=None):
    if idxes is None:
        idxes = [_ for _ in range(len(l))]
    c = 0
    for idx in idxes:
        if l[idx]:
            c += 1
    return float(c) / len(idxes)


# the input is a tuple of gold_dict, model prediction and whether to use cache
# and teh output is whether the model prediction passes the entire test suite
def judge(args: Tuple[Dict[str, Any], str, bool]) -> (bool,str):
    error_source = ''
    gold_dict, pred, use_cache = args

    testsuite_paths = gold_dict['testsuite']
    gold_query = gold_dict['query']
    order_matters = 'order by' in gold_query.lower()
    # db_path = gold_dict['db_path']
    db_path = "place_holder"

    # if already computed sometime before
    # and cache allowed, directly return the result
    k = (db_path, gold_query, pred)
    if use_cache and k in cache:
        return cache[k]

    pass_all_testcase = True
    import asyncio
    for testcase_path in sorted(testsuite_paths):

        start = time.time()
        # print("testcase_path ",testcase_path)
        flg, gold_result = asyncio.run(exec_on_db(testcase_path, gold_query, timeout=GOLD_TIMEOUT))
        duration = time.time() - start
        timeout = ADDITIVE_OVERHEAD + MULTIPLICATIVE_OVERHEAD * duration

        if flg != 'result':
            # print('Warning: executing gold query results in an exception')
            error_source = 'gold_error'
            # print("gold error")
            # print(gold_query)
            # print(gold_result)
            # print(testcase_path)
            continue
        flg, pred_result = asyncio.run(exec_on_db(testcase_path, pred, timeout=int(timeout)))
        if flg != 'result':
            # print('syntax error:',gold_dict['idx'])
            # print("gold_result ", gold_query)
            # print("pred_result ", pred)
            # print("question ",gold_dict['question'])
            # print()
            error_source = "syntax error"
            pass_all_testcase = False
            break
        if not result_eq(gold_result, pred_result, order_matters):
            # print("gold_result ", gold_result)
            # print("pred_result ", pred_result)
            # print("gold_result ", gold_query)
            # print("pred_result ", pred)
            # # print("question ", gold_dict['question'])
            # print()
            # print('value error:', gold_dict['idx'])
            error_source = "value error"
            pass_all_testcase = False
            break

    # save the results in the cache
    # if error_source=='':
    #     print("correct question ", gold_dict['question'])
    if use_cache:
        cache[k] = [pass_all_testcase,error_source]
    return [pass_all_testcase,error_source]


# cache is a dictionary
# the key is a ternary tuple (empty_database_path, SQL1, SQL2)
# the value is whether SQL1 and SQL2 are equivalent, judged by the test suites
def load_cache() -> Dict[Tuple[str, str, str], bool]:
    if os.path.exists(cache_path):
        d = m.dict(pkl.load(open(cache_path, 'rb')))
        for k, v in d.items():
            cache[k] = v
    return cache


# dump the cache
def save_cache():
    pkl.dump(dict(cache), open(cache_path, 'wb'))


def main(preds: List[str], gold_file: str = "classical_test.pkl", verbose: bool = True,
         num_processes: int = NUM_PROCESSES, subset: str = 'full', use_cache: bool = True,args=None) -> List[bool]:
    gold_dicts = pkl.load(open(gold_file, 'rb'))
    if args.selected_evaluation_file is not None:
        import json
        with open(args.selected_evaluation_file) as f:
            selected_evaluation_indices = json.load(f)
        old_gold_dicts = gold_dicts
        gold_dicts = []
        for idx in selected_evaluation_indices:
            old_gold_dicts[idx]['idx'] = idx
            gold_dicts.append(old_gold_dicts[idx])
    gold_dicts_length = len(gold_dicts)
    # import json
    # with open("/Users/hodge/Desktop/transfer/1112/full_advising_1108.json") as f:
    #     examples = json.load(f)
    # test_examples = []
    # for e in examples:
    #     # print(e.keys())
    #     if e["question-split"] in ['dev']:
    #         test_examples.append(e)
    # assert len(test_examples)==gold_dicts_length
    for i in range(gold_dicts_length):
        # print('$'*30)
        # print(gold_dicts[i]['db_path'])
        # gold_dicts[i]['db_path'] = os.path.join(args.test_suite_database_dir,gold_dicts[i]['db_path'])
        # gold_dicts[i]['question'] = test_examples[i]['question']
        old_testsuite = gold_dicts[i]['testsuite']
        # print(old_testsuite)
        gold_dicts[i]['testsuite'] = set()
        for suite in old_testsuite:
            if '/Users/hodge/Desktop/summer_research' in suite:
                print(os.path.join(args.original_database_dir, '/'.join(suite.split('/')[-2:])))
                exit(0)
                gold_dicts[i]['testsuite'].add(os.path.join(args.original_database_dir, '/'.join(suite.split('/')[-2:])))
                continue
            else:
                gold_dicts[i]['testsuite'].add(os.path.join(args.test_suite_database_dir,suite))
        # print(gold_dicts[i])
        # exit()
    # if subset != 'full':
    #     gold_dicts = [d for d in gold_dicts if subset in d['db_path']]

    #     for j in range(len(gold_dicts[i]['variables'])):
    #         gold_dicts[i]['variables'][j]['ancestor_of_occuring_column'] = None
    # if subset in ["restaurants"]:
    #     import json
    #     with open(f"./tmp/selected_evaluation_indices_{subset}.json") as f:
    #         train_indices = json.load(f)
    #     old_gold_dicts = gold_dicts
    #     gold_dicts = []
    #     for i,e in enumerate(old_gold_dicts):
    #         if i not in train_indices:
    #             gold_dicts.append(e)
    print("len(gold_dicts) ",len(gold_dicts))
    # for i in range(len(gold_dicts)):
    #     # gold_dicts[i]['text'] = None
    #     # gold_dicts[i]['query'] = None
    #     # gold_dicts[i]['variables'] = None
    #     # gold_dicts[i]['orig_id'] = None
    #     # gold_dicts[i]['db_path'] = None
    #     # gold_dicts[i]['db_id'] = None
    #     print(gold_dicts[i])
    # exit(0)
    if args.eval_num!=-1:
        gold_dicts = gold_dicts[:args.eval_num]
    assert len(gold_dicts) == len(preds), f"gold_dicts length: {len(gold_dicts)},preds length: {len(preds)}"
    group_name2idxes = defaultdict(list)

    for idx, gold_dict in enumerate(gold_dicts):
        group_name2idxes[gold_dict['db_id']].append(idx)

    with Pool(num_processes) as pool:
        old_result = list(tqdm.tqdm(pool.imap(judge, zip(gold_dicts, preds, repeat(use_cache, len(preds)))), total=len(gold_dicts)))
    # print(old_result,len(old_result))
    result = [i[0] for i in old_result if i[1]!='gold_error']
    syntax_err = 0
    value_err = 0
    for i in old_result:
        if i[1]=="syntax error":
            syntax_err += 1
        elif i[1]=="value error":
            value_err += 1
    total_error = syntax_err+value_err
    # print("verify: ",total_error/len(result))
    # if total_error==0:
    #     print("syntaxt error: ", 0)
    #     print("value error: ", 0)
    # else:
    #     print("syntaxt error: ",syntax_err,syntax_err/total_error)
    #     print("value error: ", value_err,value_err/total_error)
    # print("total error: ",total_error)
    # print("gold error: ", len(old_result)-len(result))

    # if verbose:
    evaluation_accuracy = acc(result)
    print('overall accuracy: ', evaluation_accuracy)
    import json
    with open(args.out_file,'w') as f:
        json.dump({"Acc":evaluation_accuracy},f)
        # for group, idxes in group_name2idxes.items():
        #     print('accuracy for ', group, acc(result, idxes))
    return result

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str, default='classical_test.pkl',
                        help="the path to the predicted queries")
    parser.add_argument('--pred', dest='pred', type=str, help="the path to the predicted queries")
    parser.add_argument('--out_file', type=str, required=True, help='the output file path')
    parser.add_argument('--eval_num', type=int, required=True, help='eval_num')
    parser.add_argument('--test_suite_database_dir', type=str, required=True, help='test_suite_database_dir')
    parser.add_argument('--num_processes', default=NUM_PROCESSES, help='number of processes to use')
    parser.add_argument('--subset', default='full', choices=('atis', 'kaggle','advising', 'academic', 'imdb', 'restaurants', 'geography', 'scholar', 'yelp', 'full'),
                        help='which subset to evaluate on.')
    parser.add_argument('--disable_cache', default=False, action='store_true',
                        help='whether to directly apply previously computed result and cache the current results. '
                             'use this flag to disable caching.')
    parser.add_argument('--original_database_dir', default=None,type=str, required=False, help='original_database_dir')
    parser.add_argument('--selected_evaluation_file', default=None, type=str, required=False, help='selected_evaluation_file')
    args = parser.parse_args()

    preds = load_predictions(args.pred)
    # assert not os.path.exists(args.out_file), 'output file path %s already exists' % args.out_file

    use_cache = not args.disable_cache
    if use_cache:
        load_cache()

    result = main(preds=preds, gold_file=args.gold, verbose=True, num_processes=args.num_processes,
                  subset=args.subset, use_cache=use_cache,args=args)
    # import json
    # with open(args.out_file,'w') as f:
    #     json.dump(result,f)
    # pkl.dump(result, open(args.out_file, 'wb'))
    # print('total time used: ', time.time() - start)

    if use_cache:
        save_cache()
