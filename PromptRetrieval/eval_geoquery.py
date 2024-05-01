import os
import re
import json
import argparse

def extract_program(a_string,lan='python',first_block_only=False):
    indices_object = re.finditer(pattern="```", string=a_string)
    indices = [index.start() for index in indices_object]
    contents = ''
    if len(indices) == 0:
        contents = a_string
    elif len(indices) % 2 == 0:
        for i in range(0, len(indices), 2):
            cur_str = a_string[indices[i]:indices[i + 1]]
            if cur_str.startswith(f"```{lan}"):
                cur_str = cur_str[len(f"```{lan}"):]
            elif cur_str.startswith(f"```\n{lan}"):
                cur_str = cur_str[len(f"```\n{lan}"):]
            elif cur_str.startswith("```"):
                cur_str = cur_str[len("```"):]
            contents += cur_str
            if first_block_only:
                break
    else:
        if lan=='ring':
            contents = a_string.replace(f"```{lan}", '').replace("```", '')
        else:
            contents = a_string.replace(f"```{lan}", '').replace("```", '').replace(f"{lan}\n", '')
    lines = contents.strip().split('\n')
    if lines[-1].isidentifier():
        contents = '\n'.join(lines[:-1])
    if lan=='ring':
        return contents
    return contents.replace(f"{lan}\n", '')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, type=str)
    args = parser.parse_args()

    preds = []
    args.output_dir = os.path.join(args.output_dir,'geoquery')
    for i in range(344):
        with open(os.path.join(args.output_dir,f"{i}.json")) as f:
            example = json.load(f)
            preds.append(extract_program(example['pred'].replace('\n', ' '),lan='sql').replace('SQL:','').replace('SQL',''))
    with open(os.path.join(args.output_dir,'preds_geogrpahy.txt'), 'w') as f:
        for p in preds:
            f.write(p.strip() + '\n')
    with open(os.path.join(args.output_dir,'eval_phase_selected_indices.json'),'w') as f:
        json.dump(list(range(len(preds))),f,indent=2)
    os.system(f"python3 data/test_suite_sql_eval/evaluate_classical.py "
              f"--gold=data/test_suite_database_gold_sql/gold_pkls/geography_gold.pickle "
              f"--pred={os.path.join(args.output_dir,'preds_geogrpahy.txt')} --subset=geography --out_file={os.path.join(args.output_dir,'eval_out.json')} "
              f"--test_suite_database_dir data/test_suite_database_gold_sql/test_suite_database "
              f"--eval_num -1 --disable_cache --selected_evaluation_file {os.path.join(args.output_dir,'eval_phase_selected_indices.json')} "
              f"--original_database_dir data/test_suite_database_gold_sql/spider")