import json,os,copy,transformers,torch,argparse,random
from tqdm import tqdm
import numpy as np
from utils import slot_values_to_seq_sql,table_prompt,PreviousStateRecorder,sql_pred_parse,typo_fix
from utils import evaluate
from inference import RETRIEVAL_FUNCS
from eval_geoquery import extract_program
from collections import defaultdict

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def conversion(prompt, reverse=False):
    conversion_dict = {"leaveat": "depart_time", "arriveby": "arrive_by_time",
                       "book_stay": "book_number_of_days",
                       "food": "food_type"}
    reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
    used_dict = reverse_conversion_dict if reverse else conversion_dict

    for k, v in used_dict.items():
        prompt = prompt.replace(k, v)
    return prompt

def llm_generate(pipeline_generator,prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant good at conversation."},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = pipeline_generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    formatted_prompt_ids = pipeline_generator.tokenizer(formatted_prompt)['input_ids']
    if len(formatted_prompt_ids) > 8000:
        formatted_prompt = pipeline_generator.tokenizer.decode(formatted_prompt_ids[-8000:])
    cur_batch = [formatted_prompt]
    batch_outputs = pipeline_generator(
        cur_batch,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id=pad_token_id,
        temperature=None,
        top_p=None
    )
    o = batch_outputs[0][0]["generated_text"][len(cur_batch[0]):]
    parsed_output = extract_program(o.replace('\n', ' '),lan='sql')
    if ' from ' in parsed_output.lower():
        cut_point = parsed_output.lower().index(' from ')
        parsed_output = parsed_output[cut_point+len(' from'):]
    return parsed_output.replace(';','').replace('"','').replace("'",'')

def get_instance(example_id,example):
    prompt_text = f"Example #{example_id}\n"
    last_slot_values = {s: v.split(
        '|')[0] for s, v in example['last_slot_values'].items()}
    prompt_text += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"
    last_sys_utt = example['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"Q: [user] {example['dialog']['usr'][-1]}\n"
    prompt_text += f"SQL: {conversion(slot_values_to_seq_sql(example['turn_slot_values']))};\n"
    prompt_text += "\n\n"
    return prompt_text

if __name__=='__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,default='mwoz')
    parser.add_argument('--batch_size', type=int,default=16)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--emb_batch_size', type=int, default=16)
    parser.add_argument('--retrieval', type=str,default=None)
    args = parser.parse_args()

    with open('data/mwoz_train.json') as f:
        processed_train_examples = json.load(f)
    with open('data/mwoz_eval.json') as f:
        processed_test_examples = json.load(f)
    with open('data/mw24_ontology.json') as f:
        ontology = json.load(f)

    prediction_recorder = PreviousStateRecorder()
    all_result = []
    n_total = 0
    n_correct = 0
    total_acc = 0
    total_f1 = 0

    output_dir = os.path.join(args.output_dir,'mwoz')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    retrieved_ids = None
    if args.retrieval is not None:
        retrieved_ids = RETRIEVAL_FUNCS[args.retrieval](model_id=args.retrieval,
                                                        queries=[e['history'] for e in processed_test_examples],
                                                        docs=[e['history'] for e in processed_train_examples],
                                                        task=args.task,
                                                        batch_size=args.emb_batch_size, top_k=24)
    example_id = -1
    pipeline_generator = transformers.pipeline(
        "text-generation",
        model="Qwen/CodeQwen1.5-7B-Chat",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    terminators = [
        pipeline_generator.tokenizer.eos_token_id,
        pipeline_generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pad_token_id = pipeline_generator.tokenizer.eos_token_id
    result_dict = defaultdict(list)
    for data_item in tqdm(processed_test_examples,desc=f'LLM prediction'):
        n_total += 1
        example_id += 1
        assert example_id==data_item['idx'],f"{example_id}, {data_item['idx']}"
        predicted_context = prediction_recorder.state_retrieval(data_item)
        modified_item = copy.deepcopy(data_item)
        modified_item['last_slot_values'] = predicted_context
        one_example_text = f"{data_item['name']}_{data_item['history']}"
        one_example_text = one_example_text.replace('\'', '%%%%%').replace('\"', '$$$$$')

        prompt_string = f"{conversion(table_prompt)}\n"
        selected_ids = []
        if args.retrieval is not None:
            selected_ids = retrieved_ids[example_id]
            for count,example_idx in enumerate(selected_ids):
                prompt_string += get_instance(count, processed_train_examples[example_idx])
        prompt_string += f"Example #{len(selected_ids)}\n"
        last_slot_values = predicted_context
        prompt_string += f"[context] {conversion(', '.join({f'{slot}: {value}' for slot, value in last_slot_values.items()}))}\n"
        last_sys_utt = modified_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        prompt_string += f"[system] {last_sys_utt}\n"
        prompt_string += f"Q: [user] {modified_item['dialog']['usr'][-1]}\n"
        prompt_string += "Please write the SQL.\nPlease wrap the SQL with ```."
        data_item['prompt'] = prompt_string

        completion = llm_generate(pipeline_generator=pipeline_generator,prompt=prompt_string)
        completion = conversion(completion, reverse=True)
        predicted_slot_values = {}
        try:
            predicted_slot_values = sql_pred_parse(completion)
        except Exception as e:
            data_item['not_valid'] = 1
        predicted_slot_values = typo_fix(predicted_slot_values, ontology=ontology, version=2.4)
        context_slot_values = data_item['last_slot_values']  # a dictionary
        all_slot_values = prediction_recorder.state_retrieval(data_item).copy()
        for s, v in predicted_slot_values.items():
            if s in all_slot_values and v == "[DELETE]":
                del all_slot_values[s]
            elif v != "[DELETE]":
                all_slot_values[s] = v
        all_slot_values = {k: v.split('|')[0] for k, v in all_slot_values.items()}
        prediction_recorder.add_state(data_item, all_slot_values)
        data_item['pred'] = all_slot_values
        data_item['ontology_path'] = 'data/mw24_ontology.json'
        data_item['completion'] = completion
        all_result.append(data_item)
        this_jga, this_acc, this_f1 = evaluate(all_slot_values, data_item['slot_values'])
        total_acc += this_acc
        total_f1 += this_f1
        if this_jga:
            n_correct += 1
            result_dict[data_item['turn_id']].append(1)
        else:
            result_dict[data_item['turn_id']].append(0)

    print(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}\n")
    with open(os.path.join(output_dir,'result.txt'),'w') as f:
        f.write(f"correct {n_correct}/{n_total}  =  {n_correct / n_total}\n")