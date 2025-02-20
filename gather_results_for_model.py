import glob
import json
import re
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import os

def extract_info_from_path(filepath: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract model name and checkpoint number from filepath"""
    path = Path(filepath)
    
    try:
        parts = path.parts
        saves_idx = parts.index('saves')
        lora_idx = parts.index('lora')
        
        # Get the base model name
        model_name = parts[saves_idx + 1]
        
    except (ValueError, IndexError):
        return None, None
    
    checkpoint_match = re.search(r'checkpoint-(\d+)', str(path))
    checkpoint = int(checkpoint_match.group(1)) if checkpoint_match else None
    
    return model_name, checkpoint

def extract_info(filepath: str) -> Optional[str]:
    """Extract model name from filepath using -5th index"""
    try:
        parts = Path(filepath).parts
        return parts[-5], parts[-4]
    except IndexError:
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Gather model results into a JSONL file')
    parser.add_argument('directory', type=str, help='Directory to search for JSON files')
    args = parser.parse_args()
    
    # Find all JSON files in the directory
    pattern = os.path.join(args.directory, '**/*.json')
    json_files = glob.glob(pattern, recursive=True)
    
    # Filter out unwanted files
    json_files = [f for f in json_files if 
                  'model_meta' not in f and 
                  'predictions' not in f]
    
    print(f"Found {len(json_files)} relevant JSON files")
    
    # Create output JSONL file in the same directory
    output_file = os.path.join(args.directory, 'results_summary.jsonl')
    
    with open(output_file, 'w') as outf:
        for filepath in json_files:
            model_name, dataset = extract_info(filepath)
            if not model_name:
                continue

            
                
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    scores = data["scores"]
                    if "test" in scores:
                        scores = scores["test"][0]
                    elif "standard" in scores:
                        scores = scores["standard"][0]
                    else:
                        raise ValueError(f"No test scores found in {filepath}: {scores.keys()}")

                    # if a mFollowIR dataset, also get ["og"]["ndcg_at_20"]
                    if "mFollowIR" in dataset:
                        ndcg_at_20 = scores["og"]["ndcg_at_20"]
                    else:
                        ndcg_at_20 = None

                    result = {
                        'model_name': model_name,
                        'dataset': dataset,
                        'main_score': scores["main_score"],
                        'ndcg_at_20': ndcg_at_20
                    }
                    outf.write(json.dumps(result) + '\n')
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {filepath}: {e}")
    
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()