#!/usr/bin/env python3
"""
Script to read TSV files from a compressed Common Voice dataset file (tar.gz),
decompress it into the current project folder, aggregate clip names,
compress audio files into tar archives with adaptive paths, and update metadata files.
"""

import os
import csv
import json
import tarfile
import shutil
import argparse
import tempfile
import hashlib
from pathlib import Path
from collections import defaultdict

def extract_corpus_info_from_tarball(tarball_path):
    """
    Extract the corpus information from the tarball filename.
    
    Args:
        tarball_path: Path to the tar.gz file
        
    Returns:
        Tuple containing:
        - Corpus directory name (e.g., 'cv-corpus-21.0-2025-03-14')
        - Version string (e.g., '21_0')
        - Language code (e.g., 'am' or 'hy-AM')
    """
    filename = os.path.basename(tarball_path)
    
    # Try to find the corpus name and version in the format "cv-corpus-XX.0-YYYY-MM-DD"
    import re
    corpus_match = re.search(r'(cv-corpus-(\d+\.\d+)-\d{4}-\d{2}-\d{2})', filename)
    
    if corpus_match:
        corpus_dir = corpus_match.group(1)
        # Convert "XX.0" to "XX_0" format for the version
        version = corpus_match.group(2).replace('.', '_')
        
        # Extract language code from the filename
        # Handle both simple (am) and hyphenated (hy-AM) language codes
        # Example: cv-corpus-21.0-2025-03-14-am.tar.gz -> am
        # Example: cv-corpus-21.0-2025-03-14-hy-AM.tar.gz -> hy-AM
        
        # Get the part after the corpus directory name
        remaining = filename.replace(corpus_dir, '').lstrip('-')
        # Remove the file extension
        language_part = remaining.split('.')[0]
        
        # If there's a hyphen, it might be a hyphenated language code
        if '-' in language_part:
            # For hyphenated codes like "hy-AM", use the full language part
            language_code = language_part
        else:
            # For simple codes like "am", just use the language part
            language_code = language_part
        
        return corpus_dir, version, language_code
    
    # Default values if we can't extract the information
    # Try to extract language code as a fallback
    parts = filename.split('-')
    if len(parts) > 1:
        # Get the last part before the file extension
        language_code = parts[-1].split('.')[0]
        if len(parts) > 2 and '-' in parts[-2]:
            # Check if it might be a hyphenated code
            language_code = f"{parts[-2]}-{language_code}"
    else:
        language_code = "unknown"
    
    return "cv-corpus-20.0-2024-12-06", "20_0", language_code

def is_language_already_processed(language_code, version, output_dir=None):
    """
    Check if a language has already been processed.
    
    Args:
        language_code: Language code (e.g., 'am')
        version: Version string (e.g., '20_0')
        output_dir: Custom output directory (optional)
        
    Returns:
        Boolean indicating whether the language has already been processed
    """
    # Determine the base directory
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(f"common_voice_{version}")
    
    # Check if transcripts directory exists for this language
    transcripts_dir = base_dir / "transcripts" / language_code
    if not transcripts_dir.exists():
        print(f"Transcripts directory not found: {transcripts_dir}")
        return False
    
    # Check if audio directory exists for this language
    audio_dir = base_dir / "audio" / language_code
    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return False
    
    # Check if at least one split directory exists in the audio directory
    split_dirs = ["train", "test", "dev", "validated", "invalidated", "other"]
    split_exists = False
    for split in split_dirs:
        if (audio_dir / split).exists():
            split_exists = True
            break
    
    if not split_exists:
        print(f"No split directories found in audio directory: {audio_dir}")
        return False
    
    # Check if language is in n_shards.json
    n_shards_file = base_dir / "n_shards.json"
    if n_shards_file.exists():
        try:
            with open(n_shards_file, 'r', encoding='utf-8') as f:
                n_shards = json.load(f)
                if language_code not in n_shards:
                    print(f"Language {language_code} not found in n_shards.json")
                    return False
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Error reading n_shards.json")
            return False
    else:
        print(f"n_shards.json not found: {n_shards_file}")
        return False
    
    # If all checks pass, the language has already been processed
    return True

def extract_tarball(tarball_path, extract_dir, output_dir=None):
    """
    Extract a tar.gz file to the specified directory.
    
    Args:
        tarball_path: Path to the tar.gz file
        extract_dir: Directory to extract the contents to
        output_dir: Custom output directory (optional)
        
    Returns:
        Tuple containing:
        - Path to the extracted language directory
        - Language code extracted from the filename
        - Version string extracted from the filename (e.g., '20_0')
    """
    print(f"Extracting {tarball_path} to {extract_dir}...")
    
    with tarfile.open(tarball_path, "r:gz", errorlevel=1) as tar:
        tar.extractall(path=extract_dir)
    
    # The tarball typically contains a directory like cv-corpus-21.0-2025-03-14/am
    # where 'am' is the language code. We need to find this directory.
    
    # Extract corpus directory name, version, and language code from the filename
    expected_corpus_dir, version, language_code = extract_corpus_info_from_tarball(tarball_path)
    
    # Look for the corpus directory
    corpus_dir = os.path.join(extract_dir, expected_corpus_dir)
    if not os.path.exists(corpus_dir):
        # If the expected corpus directory doesn't exist, try to find any directory that starts with "cv-corpus"
        for item in os.listdir(extract_dir):
            if item.startswith("cv-corpus"):
                corpus_dir = os.path.join(extract_dir, item)
                break
        
        if not os.path.exists(corpus_dir):
            raise FileNotFoundError(f"Could not find corpus directory in extracted contents of {tarball_path}")
    
    # Look for the language directory
    language_dir = os.path.join(corpus_dir, language_code)
    if not os.path.exists(language_dir):
        # If the language directory doesn't exist, try to find it with case-insensitive matching
        # or by looking for directories that might match the language code
        found = False
        for item in os.listdir(corpus_dir):
            # Try case-insensitive matching
            if item.lower() == language_code.lower():
                language_dir = os.path.join(corpus_dir, item)
                language_code = item  # Update the language code to match the actual directory name
                found = True
                print(f"Found language directory with case-insensitive matching: {item}")
                break
            
            # Try matching with hyphenated codes (e.g., "hy-AM" might be stored as "hy_AM" or just "hy")
            if '-' in language_code:
                base_code = language_code.split('-')[0]
                if item.lower() == base_code.lower() or item.lower() == language_code.lower().replace('-', '_'):
                    language_dir = os.path.join(corpus_dir, item)
                    language_code = item  # Update the language code to match the actual directory name
                    found = True
                    print(f"Found language directory for hyphenated code: {item}")
                    break
        
        if not found:
            # If we still can't find it, list the available directories to help with debugging
            print(f"Available directories in {corpus_dir}:")
            for item in os.listdir(corpus_dir):
                if os.path.isdir(os.path.join(corpus_dir, item)):
                    print(f"  - {item}")
            
            raise FileNotFoundError(f"Could not find language directory {language_code} in {corpus_dir}")
    
    # Move TSV files to transcripts folder
    move_tsv_files_to_transcripts(language_dir, language_code, version, output_dir)
    
    return language_dir, language_code, version

def move_tsv_files_to_transcripts(language_dir, language_code, version, output_dir=None):
    """
    Move TSV files from the extracted language directory to the transcripts folder.
    
    Args:
        language_dir: Path to the language directory
        language_code: Language code (e.g., 'ab')
        version: Version string (e.g., '20_0')
        output_dir: Custom output directory (optional)
    """
    # List of TSV files to move
    tsv_files = [
        "train.tsv",
        "invalidated.tsv",
        "other.tsv",
        "test.tsv",
        "validated.tsv",
        "dev.tsv"  # Some datasets might have this
    ]
    
    # Create transcripts directory
    if output_dir:
        transcripts_dir = Path(f"{output_dir}/transcripts/{language_code}")
    else:
        transcripts_dir = Path(f"common_voice_{version}/transcripts/{language_code}")
    
    os.makedirs(transcripts_dir, exist_ok=True)
    print(f"Created transcripts directory: {transcripts_dir}")
    
    # Copy TSV files to transcripts directory
    print("\nMoving TSV files to transcripts directory...")
    for tsv_file in tsv_files:
        source_path = os.path.join(language_dir, tsv_file)
        if os.path.exists(source_path):
            dest_path = os.path.join(transcripts_dir, tsv_file)
            shutil.copy2(source_path, dest_path)
            print(f"  Copied {tsv_file} to {dest_path}")
    
    # Also copy test.tsv to dev.tsv if dev.tsv doesn't exist (common practice in Common Voice)
    test_tsv = os.path.join(language_dir, "test.tsv")
    dev_tsv = os.path.join(language_dir, "dev.tsv")
    if os.path.exists(test_tsv) and not os.path.exists(dev_tsv):
        dev_dest = os.path.join(transcripts_dir, "dev.tsv")
        if not os.path.exists(dev_dest):
            shutil.copy2(test_tsv, dev_dest)
            print(f"  Created dev.tsv from test.tsv")

def extract_clip_paths(file_path):
    """Extract clip paths from a TSV file."""
    clip_paths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            if 'path' in row and row['path']:
                clip_paths.append(row['path'])
    
    return clip_paths

def create_tar_archives(clips, clips_dir, output_dir, language, split, clips_per_archive=40000):
    """
    Create tar archives for a list of clip paths.
    
    Args:
        clips: List of clip paths
        clips_dir: Directory containing the clip files
        output_dir: Directory to save the tar archives
        language: Language code (e.g., 'nr')
        split: Dataset split (e.g., 'train', 'test')
        clips_per_archive: Number of clips per archive
    
    Returns:
        List of created archive paths
    """
    if not clips:
        print(f"  No clips to compress for {split}")
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split clips into partitions
    partitions = [clips[i:i + clips_per_archive] for i in range(0, len(clips), clips_per_archive)]
    
    archive_paths = []
    
    for i, partition in enumerate(partitions):
        # Create archive path with adaptive naming
        archive_name = f"{language}_{split}_{i}.tar"
        archive_path = os.path.join(output_dir, archive_name)
        archive_paths.append(archive_path)
        
        print(f"  Creating archive {archive_path} with {len(partition)} clips...")
        
        # Create tar archive
        with tarfile.open(archive_path, "w") as tar:
            for clip_path in partition:
                clip_file = os.path.join(clips_dir, clip_path)
                if os.path.exists(clip_file):
                    # Add file to archive with just the filename (not the full path)
                    tar.add(clip_file, arcname=clip_path)
                else:
                    print(f"    Warning: Clip file not found: {clip_file}")
    
    return archive_paths

def update_languages_file(language_code, language_name, version, output_dir=None):
    """Update the languages.py file with the new language if it doesn't exist."""
    if output_dir:
        languages_file = Path(f"{output_dir}/languages.py")
    else:
        languages_file = Path(f"common_voice_{version}/languages.py")
    
    if not languages_file.exists():
        print(f"Warning: {languages_file} not found. Creating a new file.")
        with open(languages_file, 'w', encoding='utf-8') as f:
            f.write(f"LANGUAGES = {{'{language_code}': '{language_name}'}}\n")
        return
    
    # Read the current content
    with open(languages_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the language is already in the file
    if f"'{language_code}':" in content:
        print(f"Language {language_code} already exists in languages.py")
        return
    
    # Add the new language to the dictionary
    # This is a simple approach; a more robust approach would use AST to parse and modify the Python code
    content = content.replace("LANGUAGES = {", f"LANGUAGES = {{'{language_code}': '{language_name}', ")
    
    # Write the updated content
    with open(languages_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Added language {language_code} to languages.py")

def update_n_shards_file(language_code, split_archives, version, output_dir=None):
    """Update the n_shards.json file with the number of archives for each split."""
    if output_dir:
        n_shards_file = Path(f"{output_dir}/n_shards.json")
    else:
        n_shards_file = Path(f"common_voice_{version}/n_shards.json")
    
    # Read the current content if the file exists
    if n_shards_file.exists():
        with open(n_shards_file, 'r', encoding='utf-8') as f:
            try:
                n_shards = json.load(f)
            except json.JSONDecodeError:
                n_shards = {}
    else:
        n_shards = {}
    
    # Count the number of archives for each split
    split_counts = {}
    for split, archives in split_archives.items():
        split_counts[split] = len(archives)
    
    # Add dev split if test exists (common practice in Common Voice)
    if 'test' in split_counts:
        split_counts['dev'] = split_counts['test']
    
    # Update the n_shards dictionary
    n_shards[language_code] = split_counts
    
    # Write the updated content
    with open(n_shards_file, 'w', encoding='utf-8') as f:
        json.dump(n_shards, f, indent=4)
    
    print(f"Updated n_shards.json with {language_code} shard counts")

def compute_stats_from_tsv(language_code, split_clips, corpus_dir):
    """Compute comprehensive statistics from TSV files."""
    # Use the provided corpus directory
    tsv_dir = Path(corpus_dir) / language_code
    
    # Initialize counters and dictionaries
    total_clips = sum(len(clips) for clips in split_clips.values())
    unique_users = set()
    age_counts = defaultdict(int)
    gender_counts = defaultdict(int)
    sentence_domain_counts = defaultdict(int)
    total_duration_ms = 0
    valid_duration_secs = 0
    validated_sentences = 0
    unvalidated_sentences = 0
    reported_sentences = 0
    
    # Process each split
    for split_name, clips in split_clips.items():
        tsv_path = tsv_dir / f"{split_name}.tsv"
        if not tsv_path.exists():
            continue
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Count unique users
                if 'client_id' in row and row['client_id']:
                    unique_users.add(row['client_id'])
                
                # Count age distribution
                if 'age' in row and row['age']:
                    age_counts[row['age']] += 1
                else:
                    age_counts[''] += 1
                
                # Count gender distribution
                if 'gender' in row and row['gender']:
                    gender_counts[row['gender']] += 1
                else:
                    gender_counts[''] += 1
                
                # Count sentence domain distribution
                if 'sentence_domain' in row and row['sentence_domain']:
                    sentence_domain_counts[row['sentence_domain']] += 1
                else:
                    sentence_domain_counts[''] += 1
                
                # Count validated/unvalidated sentences
                if split_name == 'validated':
                    validated_sentences += 1
                elif split_name == 'invalidated':
                    unvalidated_sentences += 1
                
                # Count reported sentences
                if 'up_votes' in row and 'down_votes' in row:
                    if int(row.get('down_votes', 0)) > 0:
                        reported_sentences += 1
    
    # Calculate duration (estimate 5 seconds per clip)
    avg_duration_secs = 5.0
    total_duration_ms = total_clips * avg_duration_secs * 1000
    
    # Calculate valid duration (only for validated clips)
    valid_clips = len(split_clips.get('validated', []))
    valid_duration_secs = valid_clips * avg_duration_secs
    
    # Calculate hours
    total_hrs = round(total_clips * avg_duration_secs / 3600, 2)
    valid_hrs = round(valid_duration_secs / 3600, 2)
    
    # Normalize distributions to percentages
    total_with_age = sum(age_counts.values())
    age_distribution = {k: round(v / total_with_age, 2) if total_with_age > 0 else 0 for k, v in age_counts.items()}
    
    total_with_gender = sum(gender_counts.values())
    gender_distribution = {k: round(v / total_with_gender, 2) if total_with_gender > 0 else 0 for k, v in gender_counts.items()}
    
    # Ensure all expected keys exist in distributions
    for key in ['', 'twenties', 'thirties', 'teens', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties']:
        if key not in age_distribution:
            age_distribution[key] = 0
    
    for key in ['', 'male_masculine', 'female_feminine', 'transgender', 'non-binary', 'do_not_wish_to_say']:
        if key not in gender_distribution:
            gender_distribution[key] = 0
    
    # Convert defaultdict to regular dict for sentence_domain to ensure correct format
    # This preserves all domain keys found in the data without hardcoding specific ones
    sentence_domain_dict = dict(sentence_domain_counts)
    
    # Ensure it's a regular dict and not a defaultdict by creating a new dict
    sentence_domain_dict = {k: v for k, v in sentence_domain_dict.items()}
    
    # Ensure specific domain keys exist with a value of 0 if they don't exist in the data
    required_domain_keys = [
        'agriculture', 'automotive', 'finance', 'food_service_retail', 
        'general', 'healthcare', 'history_law_government', 'language_fundamentals', 
        'media_entertainment', 'nature_environment', 'news_current_affairs', 
        'technology_robotics'
    ]
    
    for key in required_domain_keys:
        if key not in sentence_domain_dict:
            sentence_domain_dict[key] = 0
    
    # Compute checksum from all clip paths
    checksum_data = ""
    for split, clips in sorted(split_clips.items()):
        for clip in sorted(clips):
            checksum_data += clip
    
    # Generate SHA-256 checksum
    checksum = hashlib.sha256(checksum_data.encode('utf-8')).hexdigest()
    
    # Create the stats dictionary with the correct format
    stats = {
        'buckets': {split: len(clips) for split, clips in split_clips.items()},
        'clips': total_clips,
        'duration': int(total_duration_ms),
        'reportedSentences': reported_sentences,
        'validatedSentences': validated_sentences,
        'unvalidatedSentences': unvalidated_sentences,
        'splits': {
            'accent': {},
            'age': age_distribution,
            'gender': gender_distribution,
            'sentence_domain': sentence_domain_dict
        },
        'users': len(unique_users),
        'size': int(total_duration_ms * 64),  # Rough estimate: 64 bytes per millisecond
        'checksum': checksum,
        'avgDurationSecs': avg_duration_secs,
        'validDurationSecs': valid_duration_secs,
        'totalHrs': total_hrs,
        'validHrs': valid_hrs
    }
    
    return stats

def parse_stats_dict(content):
    """Parse the STATS dictionary from the content of release_stats.py."""
    import ast
    import json
    
    try:
        # Extract the dictionary part (ignore the "STATS = " prefix)
        dict_start = content.find("{", content.find("STATS ="))
        if dict_start == -1:
            return None
        
        # Find the matching closing brace for the entire STATS dictionary
        brace_count = 1
        dict_end = dict_start + 1
        in_string = False
        string_char = None
        
        for i in range(dict_start + 1, len(content)):
            char = content[i]
            
            # Handle strings
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            
            # Only count braces if not in a string
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:  # We've found the closing brace
                        dict_end = i + 1
                        break
        
        # Extract the full dictionary as a string
        full_dict_str = content[dict_start:dict_end]
        
        # First try to parse as JSON (handles double quotes)
        try:
            return json.loads(full_dict_str)
        except json.JSONDecodeError:
            pass
        
        # If JSON parsing fails, try ast.literal_eval (handles single quotes)
        try:
            return ast.literal_eval(full_dict_str)
        except:
            pass
        
        # If both fail, try to normalize quotes and parse as JSON
        # Replace single quotes with double quotes, but be careful with quotes inside strings
        normalized = full_dict_str
        # This is a simple approach - replace single quotes that are likely to be JSON keys/values
        # but keep quotes that are inside string values
        import re
        # Replace single quotes around keys
        normalized = re.sub(r"'([^']+)':", r'"\1":', normalized)
        # Try parsing again
        try:
            return json.loads(normalized)
        except:
            pass
        
        return None
    except Exception as e:
        print(f"Error parsing dictionary: {e}")
        return None

def write_stats_dict(file_path, stats_dict):
    """Write the STATS dictionary to the specified file."""
    # Convert to JSON string first
    json_str = json.dumps(stats_dict, separators=(',', ':'))
    
    # Replace JSON boolean values with Python boolean values
    json_str = json_str.replace(':true', ':True')
    json_str = json_str.replace(':false', ':False')
    json_str = json_str.replace(':null', ':None')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"STATS = {json_str}\n")

def calculate_global_stats_from_locales(locales):
    """Calculate global statistics from all locale statistics."""
    global_stats = {
        'totalDuration': 0,
        'totalValidDurationSecs': 0,
        'totalHrs': 0,
        'totalValidHrs': 0
    }
    
    for language_code, stats in locales.items():
        global_stats['totalDuration'] += stats.get('duration', 0)
        global_stats['totalValidDurationSecs'] += stats.get('validDurationSecs', 0)
        global_stats['totalHrs'] += stats.get('totalHrs', 0)
        global_stats['totalValidHrs'] += stats.get('validHrs', 0)
    
    return global_stats

def validate_global_stats_consistency(full_dict):
    """Validate that global stats match the sum of all locale stats."""
    if 'locales' not in full_dict:
        return True  # Nothing to validate
    
    calculated = calculate_global_stats_from_locales(full_dict['locales'])
    
    # Check each global stat
    inconsistencies = []
    for key in ['totalDuration', 'totalValidDurationSecs', 'totalHrs', 'totalValidHrs']:
        if key in full_dict:
            actual = full_dict[key]
            expected = calculated[key]
            # Allow small rounding errors for float values
            if isinstance(actual, float) or isinstance(expected, float):
                if abs(actual - expected) > 0.01:
                    inconsistencies.append(f"{key}: actual={actual}, expected={expected}")
            else:
                if actual != expected:
                    inconsistencies.append(f"{key}: actual={actual}, expected={expected}")
    
    if inconsistencies:
        print(f"Warning: Global stats inconsistencies detected:")
        for inconsistency in inconsistencies:
            print(f"  - {inconsistency}")
        return False
    
    return True

def update_release_stats_file(language_code, split_clips, version, corpus_dir, output_dir=None):
    """Update the release_stats.py file with comprehensive statistics."""
    if output_dir:
        release_stats_file = Path(f"{output_dir}/release_stats.py")
    else:
        release_stats_file = Path(f"common_voice_{version}/release_stats.py")
    
    # Compute comprehensive statistics from TSV files
    new_stats = compute_stats_from_tsv(language_code, split_clips, corpus_dir)
    
    # Default global stats to include if creating a new file
    default_global_stats = {
        'totalDuration': 0,
        'totalValidDurationSecs': 0,
        'totalHrs': 0,
        'totalValidHrs': 0,
        'version': f'{version.replace("_", ".")}.0',
        'date': '2024-12-10',  # This should ideally be dynamically generated
        'name': f'Common Voice Corpus {version.replace("_", ".")}',
        'multilingual': True
    }
    
    # If the file doesn't exist, create it with the new stats and default global stats
    if not release_stats_file.exists():
        full_dict = {'locales': {language_code: new_stats}}
        # Add default global stats (starting at zero)
        for key, value in default_global_stats.items():
            full_dict[key] = value
        
        # Initialize global stats from the first language
        full_dict['totalDuration'] = new_stats['duration']
        full_dict['totalValidDurationSecs'] = new_stats['validDurationSecs']
        full_dict['totalHrs'] = new_stats['totalHrs']
        full_dict['totalValidHrs'] = new_stats['validHrs']
        
        write_stats_dict(release_stats_file, full_dict)
        print(f"Created release_stats.py with {language_code} stats")
        print(f"  Global stats initialized:")
        print(f"    totalDuration: {full_dict['totalDuration']}")
        print(f"    totalValidDurationSecs: {full_dict['totalValidDurationSecs']}")
        print(f"    totalHrs: {full_dict['totalHrs']}")
        print(f"    totalValidHrs: {full_dict['totalValidHrs']}")
        return
    
    # Read the current content
    with open(release_stats_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to parse the existing dictionary first before checking structure
    full_dict = parse_stats_dict(content)
    
    # If we couldn't parse the dictionary, check if it has the expected structure
    if full_dict is None:
        # Check if the file has the expected structure in a more flexible way
        if not ("STATS" in content and "=" in content and "{" in content):
            # File exists but doesn't have the expected structure, create a new one
            full_dict = {'locales': {language_code: new_stats}}
            # Add default global stats
            for key, value in default_global_stats.items():
                full_dict[key] = value
            
            # Update global stats based on the new language stats
            full_dict['totalDuration'] += new_stats['duration']
            full_dict['totalValidDurationSecs'] += new_stats['validDurationSecs']
            full_dict['totalHrs'] += new_stats['totalHrs']
            full_dict['totalValidHrs'] += new_stats['validHrs']
            
            write_stats_dict(release_stats_file, full_dict)
            print(f"Created release_stats.py with {language_code} stats and global stats (replaced invalid file)")
            return
        
        # The file has the expected structure but we couldn't parse it
        # This is likely due to a syntax error in the file
        print(f"Warning: Could not parse release_stats.py, but it appears to have the correct structure.")
        print(f"Attempting to add/update {language_code} using string manipulation.")
        
        # Try to use string manipulation to add/update the language
        try:
            # Check if the language already exists
            if f"'{language_code}':" in content or f'"{language_code}":' in content:
                # Try to update existing entry
                start_idx = content.find(f"'{language_code}':")
                if start_idx == -1:
                    start_idx = content.find(f'"{language_code}":')
                
                if start_idx != -1:
                    # Find the end of the entry
                    brace_count = 0
                    in_string = False
                    string_char = None
                    end_idx = start_idx
                    
                    for i in range(start_idx, len(content)):
                        char = content[i]
                        
                        # Handle strings
                        if char in ['"', "'"]:
                            if not in_string:
                                in_string = True
                                string_char = char
                            elif char == string_char:
                                in_string = False
                        
                        # Only count braces if not in a string
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count < 0:
                                    end_idx = i
                                    break
                        
                        # Also look for commas at the top level
                        if not in_string and brace_count == 0 and char == ',':
                            end_idx = i + 1
                            break
                    
                    # Replace the old entry with the new one
                    updated_content = content[:start_idx] + f"'{language_code}': {json.dumps(new_stats, separators=(',', ':'))}" + content[end_idx:]
                    with open(release_stats_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"Updated {language_code} stats using string manipulation")
                    print(f"Note: Global stats were not updated. Run the script on all languages to update global stats.")
                    return
            
            # If we get here, either the language doesn't exist or we couldn't find it
            # Try to add it to the locales dictionary
            locales_pos = content.find("'locales'")
            if locales_pos == -1:
                locales_pos = content.find('"locales"')
            
            if locales_pos != -1:
                # Find the opening brace of the locales dictionary
                brace_pos = content.find("{", locales_pos)
                if brace_pos != -1:
                    # Add the new entry after the opening brace
                    updated_content = content[:brace_pos+1] + f"'{language_code}': {json.dumps(new_stats, separators=(',', ':'))}, " + content[brace_pos+1:]
                    with open(release_stats_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"Added {language_code} stats using string manipulation")
                    print(f"Note: Global stats were not updated. Run the script on all languages to update global stats.")
                    return
            
            # If we get here, we couldn't find a good place to add the language
            # Create a new file as a last resort
            full_dict = {'locales': {language_code: new_stats}}
            # Add default global stats
            for key, value in default_global_stats.items():
                full_dict[key] = value
            
            # Update global stats based on the new language stats
            full_dict['totalDuration'] += new_stats['duration']
            full_dict['totalValidDurationSecs'] += new_stats['validDurationSecs']
            full_dict['totalHrs'] += new_stats['totalHrs']
            full_dict['totalValidHrs'] += new_stats['validHrs']
            
            write_stats_dict(release_stats_file, full_dict)
            print(f"Created release_stats.py with {language_code} stats and global stats (could not update existing file)")
            return
        except Exception as e:
            print(f"Error with string manipulation: {e}")
            # Create a new file as a last resort
            full_dict = {'locales': {language_code: new_stats}}
            # Add default global stats
            for key, value in default_global_stats.items():
                full_dict[key] = value
            
            # Update global stats based on the new language stats
            full_dict['totalDuration'] += new_stats['duration']
            full_dict['totalValidDurationSecs'] += new_stats['validDurationSecs']
            full_dict['totalHrs'] += new_stats['totalHrs']
            full_dict['totalValidHrs'] += new_stats['validHrs']
            
            write_stats_dict(release_stats_file, full_dict)
            print(f"Created release_stats.py with {language_code} stats and global stats (error during string manipulation)")
            return
    
    # Try to parse the existing dictionary
    full_dict = parse_stats_dict(content)
    
    if full_dict is None:
        # Couldn't parse the dictionary, create a new one
        full_dict = {'locales': {language_code: new_stats}}
        # Add default global stats
        for key, value in default_global_stats.items():
            full_dict[key] = value
        
        # Update global stats based on the new language stats
        full_dict['totalDuration'] += new_stats['duration']
        full_dict['totalValidDurationSecs'] += new_stats['validDurationSecs']
        full_dict['totalHrs'] += new_stats['totalHrs']
        full_dict['totalValidHrs'] += new_stats['validHrs']
        
        write_stats_dict(release_stats_file, full_dict)
        print(f"Created release_stats.py with {language_code} stats and global stats (replaced unparseable file)")
        return
    
    # Ensure 'locales' key exists
    if 'locales' not in full_dict:
        full_dict['locales'] = {}
    
    # Store the old stats for this language if it exists
    old_stats = full_dict['locales'].get(language_code, None)
    
    # Print current global stats before update
    print(f"\nCurrent global stats:")
    print(f"  totalDuration: {full_dict.get('totalDuration', 0)}")
    print(f"  totalValidDurationSecs: {full_dict.get('totalValidDurationSecs', 0)}")
    print(f"  totalHrs: {full_dict.get('totalHrs', 0)}")
    print(f"  totalValidHrs: {full_dict.get('totalValidHrs', 0)}")
    
    # Check if the language already exists
    if language_code in full_dict['locales']:
        current_stats = full_dict['locales'][language_code]
        
        # Check if the stats are different
        if current_stats != new_stats:
            print(f"\nUpdating {language_code} in release_stats.py")
            print(f"  Old duration: {old_stats.get('duration', 0)}, New duration: {new_stats.get('duration', 0)}")
            print(f"  Old totalHrs: {old_stats.get('totalHrs', 0)}, New totalHrs: {new_stats.get('totalHrs', 0)}")
            
            # Update global stats by removing old stats and adding new stats
            if old_stats:
                # Subtract old stats from global stats
                print(f"\nSubtracting old {language_code} stats from global totals...")
                full_dict['totalDuration'] = full_dict.get('totalDuration', 0) - old_stats.get('duration', 0)
                full_dict['totalValidDurationSecs'] = full_dict.get('totalValidDurationSecs', 0) - old_stats.get('validDurationSecs', 0)
                full_dict['totalHrs'] = full_dict.get('totalHrs', 0) - old_stats.get('totalHrs', 0)
                full_dict['totalValidHrs'] = full_dict.get('totalValidHrs', 0) - old_stats.get('validHrs', 0)
            
            # Add new stats to global stats
            print(f"Adding new {language_code} stats to global totals...")
            full_dict['totalDuration'] = full_dict.get('totalDuration', 0) + new_stats.get('duration', 0)
            full_dict['totalValidDurationSecs'] = full_dict.get('totalValidDurationSecs', 0) + new_stats.get('validDurationSecs', 0)
            full_dict['totalHrs'] = full_dict.get('totalHrs', 0) + new_stats.get('totalHrs', 0)
            full_dict['totalValidHrs'] = full_dict.get('totalValidHrs', 0) + new_stats.get('validHrs', 0)
            
            # Update the language stats
            full_dict['locales'][language_code] = new_stats
        else:
            print(f"Stats are the same, no update needed for {language_code}")
            return
    else:
        # Language doesn't exist, add it
        print(f"\nAdding new language {language_code} to release_stats.py")
        print(f"  New duration: {new_stats.get('duration', 0)}")
        print(f"  New totalHrs: {new_stats.get('totalHrs', 0)}")
        
        # Add new stats to global stats
        print(f"Adding {language_code} stats to global totals...")
        full_dict['totalDuration'] = full_dict.get('totalDuration', 0) + new_stats.get('duration', 0)
        full_dict['totalValidDurationSecs'] = full_dict.get('totalValidDurationSecs', 0) + new_stats.get('validDurationSecs', 0)
        full_dict['totalHrs'] = full_dict.get('totalHrs', 0) + new_stats.get('totalHrs', 0)
        full_dict['totalValidHrs'] = full_dict.get('totalValidHrs', 0) + new_stats.get('validHrs', 0)
        
        # Add the language stats
        full_dict['locales'][language_code] = new_stats
    
    # Ensure all global stats exist
    for key, value in default_global_stats.items():
        if key not in full_dict:
            full_dict[key] = value
    
    # Print updated global stats
    print(f"\nUpdated global stats:")
    print(f"  totalDuration: {full_dict.get('totalDuration', 0)}")
    print(f"  totalValidDurationSecs: {full_dict.get('totalValidDurationSecs', 0)}")
    print(f"  totalHrs: {full_dict.get('totalHrs', 0)}")
    print(f"  totalValidHrs: {full_dict.get('totalValidHrs', 0)}")
    
    # Validate global stats consistency
    if not validate_global_stats_consistency(full_dict):
        print("\nRecalculating global stats to fix inconsistencies...")
        calculated_global = calculate_global_stats_from_locales(full_dict['locales'])
        for key, value in calculated_global.items():
            full_dict[key] = value
        print(f"  Fixed global stats:")
        print(f"    totalDuration: {full_dict['totalDuration']}")
        print(f"    totalValidDurationSecs: {full_dict['totalValidDurationSecs']}")
        print(f"    totalHrs: {full_dict['totalHrs']}")
        print(f"    totalValidHrs: {full_dict['totalValidHrs']}")
    
    # Write the updated dictionary back to the file
    try:
        write_stats_dict(release_stats_file, full_dict)
        print(f"\nSuccessfully updated release_stats.py with {language_code} stats and global stats")
    except Exception as e:
        print(f"Error writing to release_stats.py: {e}")
        # Fallback to simple string manipulation if we can't write the full dictionary
        try:
            if language_code in full_dict.get('locales', {}):
                # Try to update existing entry
                locales_pos = content.find("'locales': {") + len("'locales': {")
                start_idx = content.find(f"'{language_code}':")
                if start_idx != -1:
                    end_idx = content.find(",", start_idx)
                    if end_idx == -1:  # This might be the last entry
                        end_idx = content.find("}", start_idx)
                    
                    # Replace the old entry with the new one
                    updated_content = content[:start_idx] + f"'{language_code}': {json.dumps(new_stats, separators=(',', ':'))}" + content[end_idx:]
                    with open(release_stats_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    print(f"Updated {language_code} stats using string manipulation")
                    print(f"Note: Global stats were not updated. Run the script on all languages to update global stats.")
            else:
                # Try to add new entry
                locales_pos = content.find("'locales': {") + len("'locales': {")
                updated_content = content[:locales_pos] + f"'{language_code}': {json.dumps(new_stats, separators=(',', ':'))}, " + content[locales_pos:]
                with open(release_stats_file, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"Added {language_code} stats using string manipulation")
                print(f"Note: Global stats were not updated. Run the script on all languages to update global stats.")
        except Exception as nested_e:
            print(f"Error with fallback string manipulation: {nested_e}")

def get_language_name(language_code):
    """Get the language name for a given language code."""
    # This is a simplified mapping; in a real scenario, you might want to use a more complete mapping
    language_map = {
        'ab': 'Abkhaz',
        'ar': 'Arabic',
        'ca': 'Catalan',
        'cs': 'Czech',
        'cy': 'Welsh',
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'et': 'Estonian',
        'eu': 'Basque',
        'fa': 'Persian',
        'fr': 'French',
        'it': 'Italian',
        'ja': 'Japanese',
        'nl': 'Dutch',
        'nr': 'IsiNdebele (South)',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'sv': 'Swedish',
        'tr': 'Turkish',
        'zh': 'Chinese',
    }
    
    return language_map.get(language_code, f"Unknown ({language_code})")

def process_language_data(language_dir, language_code, version, language_name=None, output_dir=None):
    """
    Process the language data from the extracted directory.
    
    Args:
        language_dir: Path to the language directory
        language_code: Language code (e.g., 'ab')
        version: Version string (e.g., '20_0')
        language_name: Language name (e.g., 'Abkhaz')
        output_dir: Custom output directory (optional)
    """
    if language_name is None:
        language_name = get_language_name(language_code)
    
    clips_dir = os.path.join(language_dir, "clips")
    
    # Output directory for compressed archives
    if output_dir:
        output_base_dir = Path(f"{output_dir}/audio/{language_code}")
    else:
        output_base_dir = Path(f"common_voice_{version}/audio/{language_code}")
    
    # Get the corpus directory (parent of language_dir)
    corpus_dir = os.path.dirname(language_dir)
    
    # List of TSV files to read (as specified in the task)
    tsv_files = [
        "train.tsv",
        "invalidated.tsv",
        "other.tsv",
        "test.tsv",
        "dev.tsv",
        "validated.tsv"
    ]
    
    # Dictionary to store clip paths for each split
    split_clips = defaultdict(list)
    
    # Process each file
    for tsv_file in tsv_files:
        split_name = os.path.splitext(tsv_file)[0]  # Remove .tsv extension
        file_path = os.path.join(language_dir, tsv_file)
        
        if os.path.exists(file_path):
            print(f"Processing {tsv_file}...")
            clip_paths = extract_clip_paths(file_path)
            split_clips[split_name] = clip_paths
            print(f"  Found {len(clip_paths)} clips")
        else:
            print(f"File not found: {file_path}")
    
    # Print summary
    print("\nSummary of clips per split:")
    for split_name, clips in split_clips.items():
        print(f"{split_name}: {len(clips)} clips")
    
    # Create tar archives for each split
    print("\nCreating tar archives...")
    split_archives = {}
    
    for split_name, clips in split_clips.items():
        # Create output directory for this split
        split_output_dir = output_base_dir / split_name
        
        # Create tar archives
        archives = create_tar_archives(
            clips=clips,
            clips_dir=clips_dir,
            output_dir=split_output_dir,
            language=language_code,
            split=split_name,
            clips_per_archive=40000  # Adjust this value as needed
            # clips_per_archive=10,
        )
        
        split_archives[split_name] = archives
    
    # Print summary of created archives
    print("\nCreated archives:")
    all_archives = []
    for split, archives in split_archives.items():
        all_archives.extend(archives)
        for archive in archives:
            print(f"  {archive}")
    
    # Update metadata files
    print("\nUpdating metadata files...")
    update_languages_file(language_code, language_name, version, output_dir)
    update_n_shards_file(language_code, split_archives, version, output_dir)
    update_release_stats_file(language_code, split_clips, version, corpus_dir, output_dir)
    
    # TSV files are already moved to transcripts directory during extraction
    
    print("\nProcessing completed successfully!")

def find_tarballs(directory):
    """
    Find all tar.gz files in the specified directory.
    
    Args:
        directory: Directory to search for tar.gz files
        
    Returns:
        List of paths to tar.gz files
    """
    tarballs = []
    for file in os.listdir(directory):
        if file.endswith(".tar.gz"):
            tarballs.append(os.path.join(directory, file))
    return tarballs

def main():
    parser = argparse.ArgumentParser(description="Process Common Voice dataset tar.gz files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tarball", help="Path to a single tar.gz file (e.g., cv-corpus-20.0-2024-12-06-af.tar.gz)")
    group.add_argument("--directory", help="Path to a directory containing tar.gz files")
    parser.add_argument("--language-name", help="Language name (override automatic detection)")
    parser.add_argument("--extract-dir", default=".", help="Directory to extract the tarball to (default: current directory)")
    parser.add_argument("--output-dir", help="Custom output directory (overrides the default common_voice_XX_0 directory)")
    parser.add_argument("--force", action="store_true", help="Force processing even if the language has already been processed")
    
    args = parser.parse_args()
    
    # Create the extraction directory if it doesn't exist
    extract_dir = Path(args.extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Determine which tarballs to process
    tarballs = []
    if args.tarball:
        tarballs = [args.tarball]
    elif args.directory:
        tarballs = find_tarballs(args.directory)
        if not tarballs:
            print(f"No tar.gz files found in directory: {args.directory}")
            return
    
    # Process each tarball
    for tarball in tarballs:
        try:
            print(f"\n{'='*80}")
            print(f"Processing tarball: {tarball}")
            print(f"{'='*80}\n")
            
            # Extract corpus information from the tarball filename
            _, version, language_code = extract_corpus_info_from_tarball(tarball)
            
            # Check if the language has already been processed
            if not args.force and is_language_already_processed(language_code, version, args.output_dir):
                print(f"Language {language_code} (version {version}) has already been processed. Skipping...")
                print(f"Use --force to reprocess this language if needed.")
                continue
            
            # Extract the tarball to the specified directory
            language_dir, language_code, version = extract_tarball(tarball, extract_dir, args.output_dir)
            
            # Process the language data
            language_name = args.language_name or get_language_name(language_code)
            process_language_data(language_dir, language_code, version, language_name, args.output_dir)
            
        except Exception as e:
            print(f"Error processing tarball {tarball}: {e}")
            # Continue with the next tarball instead of raising the exception

if __name__ == "__main__":
    main()
