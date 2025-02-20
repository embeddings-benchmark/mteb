#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-3250-full" -d BrightRetrieval -s biology -n 4
python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-3500-full" -d BrightRetrieval -s biology -n 4
python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-3750-full" -d BrightRetrieval -s biology -n 4



