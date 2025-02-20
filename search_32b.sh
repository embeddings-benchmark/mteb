#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-3500-full" --num_gpus 4
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-3750-full" --num_gpus 4
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2-short/lora/sft/checkpoint-1750-full" --num_gpus 4
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2-short/lora/sft/checkpoint-2000-full" --num_gpus 4

