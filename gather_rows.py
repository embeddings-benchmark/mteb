import json
import os
from glob import glob
from collections import defaultdict

class BenchmarkProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def find_results_files(self):
        """Find all results_summary.jsonl files in the directory tree."""
        return glob(os.path.join(self.root_dir, "**/results_summary.jsonl"), recursive=True)

    def parse_jsonl(self, file_path):
        """Parse JSONL file and return list of entries."""
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f if line.strip()]

class BrightProcessor(BenchmarkProcessor):
    def _is_prompted_dataset(self, dataset):
        """Check if dataset name contains a hash indicating it's a prompted version."""
        return len(dataset.split('_')) > 2 and len(dataset.split('_')[-1]) == 64

    def _get_base_dataset_name(self, dataset):
        """Get the base dataset name without the hash."""
        if self._is_prompted_dataset(dataset):
            return '_'.join(dataset.split('_')[:-1])
        return dataset

    def extract_scores(self, entries, prompted=False):
        dataset_mapping = {
            'BrightRetrieval_biology': 0,
            'BrightRetrieval_earth': 1,
            'BrightRetrieval_economics': 2,
            'BrightRetrieval_psychology': 3,
            'BrightRetrieval_robotics': 4,
            'BrightRetrieval_stackoverflow': 5,
            'BrightRetrieval_sustainability': 6,
            'BrightRetrieval_leetcode': 7,
            'BrightRetrieval_pony': 8,
            'BrightRetrieval_aops': 9,
            'BrightRetrieval_theoremqa': 10,
            'BrightRetrieval_theoremtelling': 11
        }
        
        scores = ['-'] * 12
        valid_scores = []
        
        for entry in entries:
            dataset = entry.get('dataset')
            is_prompted = self._is_prompted_dataset(dataset)
            
            # Skip if prompted status doesn't match what we want
            if prompted != is_prompted:
                continue
                
            base_dataset = self._get_base_dataset_name(dataset)
            if base_dataset in dataset_mapping:
                score = entry.get('main_score')
                if score is not None:
                    score = score * 100
                    pos = dataset_mapping[base_dataset]
                    scores[pos] = f"{score:.1f}"
                    valid_scores.append(score)
        
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        return scores, f"{avg:.1f}"

    def format_row(self, model_name, scores, avg):
        scores_str = " & ".join(str(score) for score in scores)
        return f"{model_name} & {scores_str} & {avg} \\\\"

class BeirProcessor(BenchmarkProcessor):
    def _is_prompted_dataset(self, dataset):
        """Check if dataset name contains a hash indicating it's a prompted version."""
        return len(dataset.split('_')) > 1 and len(dataset.split('_')[-1]) == 64

    def _get_base_dataset_name(self, dataset):
        """Get the base dataset name without the hash."""
        if self._is_prompted_dataset(dataset):
            return '_'.join(dataset.split('_')[:-1])
        return dataset

    def extract_scores(self, entries, prompted=False):
        dataset_mapping = {
            'arguana': 0,
            'clim-fever': 1,
            'dbpedia': 2,
            'fiqa': 3,
            'nfcorpus': 4,
            'scidocs': 5,
            'SciFact': 6,
            'touche20': 7,
            'TRECCOVID': 8
        }
        
        scores = ['-'] * 9
        valid_scores = []
        
        for entry in entries:
            dataset = entry.get('dataset')
            is_prompted = self._is_prompted_dataset(dataset)
            
            # Skip if prompted status doesn't match what we want
            if prompted != is_prompted:
                continue
                
            base_dataset = self._get_base_dataset_name(dataset)
            if base_dataset in dataset_mapping:
                score = entry.get('main_score')
                if score is not None:
                    score = score * 100
                    pos = dataset_mapping[base_dataset]
                    scores[pos] = f"{score:.1f}"
                    valid_scores.append(score)
        
        avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        return scores, f"{avg:.1f}"

    def format_row(self, model_name, scores, avg):
        scores_str = " & ".join(str(score) for score in scores)
        return f"{model_name} & {scores_str} & {avg} \\\\"

class NevIRProcessor(BenchmarkProcessor):
    def extract_scores(self, entries):
        nevir_score = None
        for entry in entries:
            if entry.get('dataset') == 'NevIR':
                score = entry.get('main_score')
                if score is not None:
                    nevir_score = score * 100
                break
        
        return [f"{nevir_score:.1f}"] if nevir_score is not None else ['-']

    def format_row(self, model_name, scores, _):
        score = scores[0]
        return f"& {model_name} & {score} \\\\"

class MFollowIRProcessor(BenchmarkProcessor):
    def extract_scores(self, entries, cross_lingual=False):
        dataset_mapping = {
            # Regular mFollowIR datasets
            'mFollowIR_fas': ('Persian', 0),
            'mFollowIR_zho': ('Chinese', 1),
            'mFollowIR_rus': ('Russian', 2),
            # Cross-lingual datasets
            'mFollowIRCrossLingual_eng-fas': ('Persian', 0),
            'mFollowIRCrossLingual_eng-zho': ('Chinese', 1),
            'mFollowIRCrossLingual_eng-rus': ('Russian', 2),
        }
        
        scores = {}
        for entry in entries:
            dataset = entry.get('dataset')
            is_cross_lingual = dataset.startswith('mFollowIRCrossLingual')
            
            # Skip if cross_lingual status doesn't match what we want
            if cross_lingual != is_cross_lingual:
                continue
                
            if dataset in dataset_mapping:
                lang, _ = dataset_mapping[dataset]
                if entry.get('ndcg_at_20') is not None:
                    ndcg = entry.get('ndcg_at_20')  # Keep as 0-1 scale
                    mrr = entry.get('main_score', 0) * 100  # Convert to percentage
                    scores[lang] = (ndcg, mrr)
        
        return scores

    def format_row(self, model_name, scores, _):
        row_parts = [f"& {model_name}"]
        
        total_ndcg = 0
        total_mrr = 0
        count = 0
        
        # Column order: Persian, Chinese, Russian
        for lang in ['Persian', 'Chinese', 'Russian']:
            row_parts.append("&")
            if lang in scores:
                ndcg, mrr = scores[lang]
                row_parts.append(f"{ndcg:.3f}")  # 3 decimal places for ndcg (0-1 scale)
                row_parts.append("& &")  # Extra & between ndcg and mrr
                row_parts.append(f"{mrr:.1f}")  # 1 decimal place for mrr (percentage)
                total_ndcg += ndcg
                total_mrr += mrr
                count += 1
            else:
                row_parts.extend(["-", "&", "-"])
            
            # Add & between language groups if not the last one
            # if lang != 'Russian':
        
        # Add averages
        if count > 0:
            avg_ndcg = total_ndcg / count
            avg_mrr = total_mrr / count
            row_parts.append("&")  # & before averages
            row_parts.append(f"{avg_ndcg:.3f}")  # 3 decimals for ndcg average (0-1 scale)
            row_parts.append("&")
            row_parts.append(f"{avg_mrr:.1f}")  # 1 decimal for mrr average (percentage)
        else:
            row_parts.extend(["&", "-", "-"])
            
        return " ".join(row_parts) + " \\\\"

def process_benchmarks(root_dir):
    """Process all results files and generate LaTeX rows for each benchmark."""
    processors = {
        'bright': BrightProcessor(root_dir),
        'beir': BeirProcessor(root_dir),
        'nevir': NevIRProcessor(root_dir),
        'mfollowir': MFollowIRProcessor(root_dir),
    }
    
    # Initialize results dictionary for each processor and variant
    results = {
        'bright': {'prompted': [], 'unprompted': []},
        'beir': {'prompted': [], 'unprompted': []},
        'nevir': [],
        'mfollowir': [],
        'mfollowir_crosslingual': []
    }
    
    # Process each results file
    for file_path in processors['bright'].find_results_files():
        entries = processors['bright'].parse_jsonl(file_path)
        if not entries:
            continue
            
        model_name = entries[0].get('model_name', 'Unknown')
        
        # Process with each processor
        for name, processor in processors.items():
            if name in ['bright', 'beir']:
                # Process both prompted and unprompted versions
                for prompted in [True, False]:
                    scores, avg = processor.extract_scores(entries, prompted=prompted)
                    row = processor.format_row(model_name, scores, avg)
                    results[name]['prompted' if prompted else 'unprompted'].append(row)
            elif name == 'nevir':
                scores = processor.extract_scores(entries)
                row = processor.format_row(model_name, scores, None)
                results[name].append(row)
            elif name == 'mfollowir':
                # Process both regular and cross-lingual versions
                scores = processor.extract_scores(entries, cross_lingual=False)
                if scores:  # Only add row if we have scores
                    row = processor.format_row(model_name, scores, None)
                    results['mfollowir'].append(row)
                
                scores = processor.extract_scores(entries, cross_lingual=True)
                if scores:  # Only add row if we have scores
                    row = processor.format_row(model_name, scores, None)
                    results['mfollowir_crosslingual'].append(row)
    
    # Write results to files
    for name, result in results.items():
        if name in ['bright', 'beir']:
            # Write prompted and unprompted versions
            for variant, rows in result.items():
                if rows:  # Only write file if we have results
                    filename = f'{name}_{variant}_rows.tex'
                    with open(filename, 'w') as f:
                        f.write("\n".join(rows))
                    print(f"Created {filename} with {len(rows)} rows")
        else:
            # Write single version for nevir and mfollowir
            if result:  # Only write file if we have results
                filename = f'{name}_rows.tex'
                with open(filename, 'w') as f:
                    f.write("\n".join(result))
                print(f"Created {filename} with {len(result)} rows")

if __name__ == "__main__":
    root_dir = "."
    process_benchmarks(root_dir)