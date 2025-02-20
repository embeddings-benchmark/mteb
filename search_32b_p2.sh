#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-1750-full" --num_gpus 2
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-2000-full" --num_gpus 2


# short version
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2-short/lora/sft/checkpoint-250-full" --num_gpus 2
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2-short/lora/sft/checkpoint-500-full" --num_gpus 2 
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2-short/lora/sft/checkpoint-750-full" --num_gpus 2
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2/lora/sft/checkpoint-2500-full" --num_gpus 2
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2-short/lora/sft/checkpoint-1000-full" --num_gpus 2
python run_mteb_no_sub.py -d "NevIR" -c "/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-32b-v2-short/lora/sft/checkpoint-1250-full" --num_gpus 2



