#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-750-full" -d BrightRetrieval -s biology -n 2
python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-500-full" -d BrightRetrieval -s biology -n 2
python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-2750-full" -d BrightRetrieval -s biology -n 2

python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-1000-full" -d BrightRetrieval -s biology -n 2
python run_mteb.py -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-1750-full" -d BrightRetrieval -s biology -n 2