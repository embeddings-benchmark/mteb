import glob
import json
import re
import matplotlib.pyplot as plt
from pathlib import Path
from ast import literal_eval

def extract_score_from_log(filepath):
   # get all the lines
    lines = open(filepath, 'r').readlines()
    for line in reversed(lines):
        if "Scores:" in line:
            # Extract the JSON part
            json_str = line.split("Scores: ")[1].strip().replace("'", '"')
            # get the part from "main_score": 0.618944323933478}}'
            json_str = json_str.split("main_score\":")[1].split("}}")[0]
            return float(json_str)

    return None

def extract_checkpoint_info(filepath):
    # Extract checkpoint number and name from filepath
    filename = Path(filepath).name
    # Updated regex to make the name part optional
    match = re.search(r'(.*)?-?checkpoint-(\d+)', filename)
    if match:
        name = match.group(1) if match.group(1) else "default"  # Use "default" if no name present
        checkpoint = int(match.group(2))
        return name, checkpoint
    return None, None

def main():
    # Find all log files
    log_files = list(glob.glob("/home/oweller2/my_scratch/mteb/*.log"))
    print(f"There are {len(log_files)} log files")
    
    # Dictionary to store results by model name
    results = {}
    
    for filepath in log_files:
        score = extract_score_from_log(filepath)
        name, checkpoint = extract_checkpoint_info(filepath)
        
        if score is not None and name is not None:
            if name not in results:
                results[name] = {'checkpoints': [], 'scores': []}
            results[name]['checkpoints'].append(checkpoint)
            results[name]['scores'].append(score)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        # Sort by checkpoint number
        checkpoints, scores = zip(*sorted(zip(data['checkpoints'], data['scores'])))
        plt.plot(checkpoints, scores, marker='o', label=name)
    
    plt.xlabel('Checkpoint')
    plt.ylabel('Main Score')
    plt.title('Model Performance Across Checkpoints')
    plt.legend()
    plt.grid(True)
    plt.savefig('nevir_scores_qwen.png')
    plt.close()

if __name__ == "__main__":
    main()
