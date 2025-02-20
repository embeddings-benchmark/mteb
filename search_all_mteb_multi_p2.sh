#!/bin/bash
#  for dataset and prompt combinations run the models
models=("llama-8b-v2")
for model in ${models[@]}; do
    python search_all_mteb.py --model_name $model --dataset_name BrightRetrieval --subtask biology
done

# python run_mteb_no_sub.py -c /home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-7b-v2/lora/sft/checkpoint-6000-full -d MiniDL19 -p "I am interested in documents which 'provide substantial information on the query'"