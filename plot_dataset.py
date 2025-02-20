import glob
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
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
        
        # Check if the path contains "prompted"
        checkpoint_part = str(path)
        if "prompted" in checkpoint_part:
            model_name = f"{model_name}_prompted"
    except (ValueError, IndexError):
        return None, None
    
    checkpoint_match = re.search(r'checkpoint-(\d+)', str(path))
    checkpoint = int(checkpoint_match.group(1)) if checkpoint_match else None
    
    return model_name, checkpoint

def get_metric_groups() -> Dict[str, List[str]]:
    """Define groups of metrics to plot together"""
    return {
        'main': ['main_score'],
        'ndcg': [f'ndcg_at_{k}' for k in (1, 5, 10)],
        'precision': [f'precision_at_{k}' for k in (1, 5, 10)],
        'recall': [f'recall_at_{k}' for k in (1, 5, 10)],
        'binary_accuracy': ['TotalBinaryAccuracy@1', 'TotalBinaryAccuracy@10', 'TotalBinaryAccuracy@1000']
    }

def create_plot(
    results: Dict[str, Dict[str, list]],
    metric: str,
    dataset: str,
    output_dir: str
) -> None:
    """Create and save a plot for a specific metric"""
    plt.figure(figsize=(12, 7))
    
    for name, data in results.items():
        if data['checkpoints'] and metric in data['scores']:
            checkpoints = data['checkpoints']
            scores = data['scores'][metric]
            if scores:  # Only plot if we have data for this metric
                plt.plot(checkpoints, scores, marker='o', label=name)
    
    plt.xlabel('Checkpoint')
    plt.ylabel(metric)
    plt.title(f'Model Performance on {dataset} Dataset - {metric}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{dataset}_{metric}_scores.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot model performance across checkpoints')
    parser.add_argument('dataset', type=str, help='Name of the dataset (e.g., MiniDL19)')
    parser.add_argument('-o', '--output-dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all matching JSON files recursively
    pattern = f'results/home/**/saves/**/lora/**/{args.dataset}.json'
    json_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(json_files)} JSON files for dataset {args.dataset}")
    
    # Dictionary to store results by model name
    results: Dict[str, Dict[str, dict]] = {}
    metric_groups = get_metric_groups()
    all_metrics = [m for group in metric_groups.values() for m in group]
    
    # Process all files
    for filepath in json_files:
        model_name, checkpoint = extract_info_from_path(filepath)
        
        if model_name and checkpoint:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)["scores"]
                    data = data["test"][0] if "test" in data else data["standard"][0]
                
                if model_name not in results:
                    results[model_name] = {
                        'checkpoints': [],
                        'scores': {metric: [] for metric in all_metrics}
                    }
                
                # Only add checkpoint once
                if checkpoint not in results[model_name]['checkpoints']:
                    results[model_name]['checkpoints'].append(checkpoint)
                    
                    # Add scores for all available metrics
                    for metric in all_metrics:
                        score = data.get(metric)
                        if score is not None:
                            results[model_name]['scores'][metric].append(score)
                        else:
                            results[model_name]['scores'][metric].append(None)
                            
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {filepath}: {e}")
    
    # Sort checkpoints and scores for each model
    for model_data in results.values():
        if model_data['checkpoints']:
            # Get sorting indices based on checkpoints
            sort_indices = sorted(range(len(model_data['checkpoints'])), 
                                key=lambda k: model_data['checkpoints'][k])
            
            # Sort checkpoints
            model_data['checkpoints'] = [model_data['checkpoints'][i] for i in sort_indices]
            
            # Sort scores for each metric
            for metric in all_metrics:
                if model_data['scores'][metric]:
                    model_data['scores'][metric] = [model_data['scores'][metric][i] 
                                                  for i in sort_indices]
    
    # Create plots for each metric group
    for group_name, metrics in metric_groups.items():
        for metric in metrics:
            # Check if any model has data for this metric
            has_data = any(bool(data['scores'].get(metric)) for data in results.values())
            if has_data:
                create_plot(results, metric, args.dataset, args.output_dir)
            else:
                print(f"No data found for metric: {metric}")

if __name__ == "__main__":
    main()