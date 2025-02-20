import subprocess
import time
import os
from typing import List, Tuple
import argparse
from multiprocessing import Process, JoinableQueue, Manager
import signal
import sys

def generate_job_commands(checkpoints: List[int], 
                         model_names: List[str], 
                         base_path: str = "/home/oweller2/my_scratch/LLaMA-Factory/saves",
                         dataset_prompt: str = None,
                         dataset_name: str = None,
                         dataset_subtask: str = None
                         ) -> List[Tuple[str, str]]:
    """Generate all job commands and their corresponding output names."""
    jobs = []
    for name in model_names:
        for checkpoint in checkpoints:
            model_path = f"{base_path}/{name}/lora/sft/checkpoint-{checkpoint}-full"
            clean_model_name = f"{name}-checkpoint-{checkpoint}-full"
            
            # # Skip if output file already exists
            # if os.path.exists(f"{clean_model_name}.log"):
            #     print(f"Skipping {clean_model_name} because it already exists")
            #     continue

            if dataset_subtask:
                command = [
                    "python",
                    "run_mteb.py",
                    f"-d {dataset_name.strip()}",
                    f"-c {model_path.strip()}",
                    f"-s {dataset_subtask.strip()}"
                ]
                if dataset_prompt:
                    command.append(f'-p "{dataset_prompt.strip()}"')
            else:
                    
                command = [
                    "python",
                    "run_mteb_no_sub.py",
                    f"-d {dataset_name.strip()}",
                    f"-c {model_path.strip()}",
                ]
                if dataset_prompt:
                    command.append(f'-p "{dataset_prompt.strip()}"')
            # print(command)
            jobs.append((command, clean_model_name))
    return jobs

def run_job(command: List[str], output_name: str, gpu_id: int, dataset_name: str, prompted: str):
    """Run a single job on the specified GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Starting job {output_name} on GPU {gpu_id}")
    with open(f"logs/{output_name}_{dataset_name}{prompted}.log", "w") as f:
        try:
            process = subprocess.Popen(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                shell=False
            )
            return_code = process.wait()
            print(f"Completed job {output_name} on GPU {gpu_id} with return code {return_code}")
            return return_code
        except Exception as e:
            print(f"Error running job {output_name} on GPU {gpu_id}: {str(e)}")
            return 1

def gpu_worker(gpu_id: int, job_queue: JoinableQueue, active_processes, dataset_name: str, prompted: str):
    """Worker process to handle jobs for a specific GPU."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore keyboard interrupts in worker processes
    
    while True:
        job = job_queue.get()
        if job is None:  # Poison pill to stop the worker
            job_queue.task_done()
            break
            
        command, output_name = job
        active_processes[gpu_id] = output_name  # Track currently running job
        
        return_code = run_job(command, output_name, gpu_id, dataset_name, prompted)
        
        active_processes[gpu_id] = None  # Clear currently running job
        job_queue.task_done()

def signal_handler(signum, frame):
    """Handle cleanup on interrupt signals."""
    print("\nInterrupt received. Cleaning up...")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="GPU Job Queue Manager")
    parser.add_argument("--model_name", type=str, default="negs-instruct", help="Model name")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for job slice")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index for job slice")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--dataset_prompt", type=str, default=None, help="Dataset prompt")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name")
    parser.add_argument("--subtask", type=str, default=None, help="Subtask name")
    args = parser.parse_args()

    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.dataset_name is None:
        raise ValueError("Dataset name is required")

    # find checkpoints for model
    checkpoints = [f for f in os.listdir(f"/home/oweller2/my_scratch/LLaMA-Factory/saves/{args.model_name}/lora/sft") if f.startswith("checkpoint-") and "full" in f]
    checkpoints = set([int(f.split("-")[1]) for f in checkpoints])
    # remove any checkpoint greater than 4k
    # checkpoints = [c for c in checkpoints if c <= 4000]
    # sort the checkpoints
    checkpoints = sorted(list(checkpoints), reverse=True)
    print(f"Found {len(checkpoints)} checkpoints for {args.model_name}: {checkpoints}")
    # Generate all jobsnegs
    all_jobs = generate_job_commands(checkpoints, [args.model_name], dataset_prompt=args.dataset_prompt, dataset_name=args.dataset_name.strip(), dataset_subtask=args.subtask)
    
    # Slice jobs based on start_idx and end_idx
    if args.end_idx is not None:
        all_jobs = all_jobs[args.start_idx:args.end_idx]
    else:
        all_jobs = all_jobs[args.start_idx:]

    # Create job queue and shared process dictionary
    job_queue = JoinableQueue()
    with Manager() as manager:
        # Dictionary to track active processes
        active_processes = manager.dict()
        for i in range(args.num_gpus):
            active_processes[i] = None

        # Create worker processes
        processes = []
        for gpu_id in range(args.num_gpus):
            dataset_name = args.dataset_name.strip()
            if args.subtask:
                dataset_name = f"{dataset_name}_{args.subtask.strip()}"
            p = Process(target=gpu_worker, args=(gpu_id, job_queue, active_processes, dataset_name, "_no" if args.dataset_prompt is None else "_yes"))
            p.start()
            processes.append(p)

        # Add jobs to queue
        for job in all_jobs:
            job_queue.put(job)

        # Add poison pills to stop workers
        for _ in range(args.num_gpus):
            job_queue.put(None)

        try:
            # Wait for all jobs to complete
            job_queue.join()
            
            # Wait for all processes to finish
            for p in processes:
                p.join()
                
        except KeyboardInterrupt:
            print("\nInterrupt received. Stopping workers...")
            # Clean termination of worker processes
            for p in processes:
                p.terminate()
                p.join()
            
            # Print status of running jobs
            for gpu_id, job_name in active_processes.items():
                if job_name:
                    print(f"Interrupted job on GPU {gpu_id}: {job_name}")

if __name__ == "__main__":
    main()