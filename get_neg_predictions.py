import json

def check_preferences(data):
    issues = []
    has_issues = 0
    
    for query_id in data:
        # Skip entries that don't match our expected pattern
        if not (query_id.endswith('_q1') or query_id.endswith('_q2')):
            continue
            
        base_id = query_id[:-3]  # Remove _q1 or _q2
        doc1_key = f"{base_id}_doc1"
        doc2_key = f"{base_id}_doc2"
        
        scores = data[query_id]
        doc1_score = scores[doc1_key]
        doc2_score = scores[doc2_key]
        
        # For q1, doc1 should be preferred (have higher score)
        if query_id.endswith('_q1'):
            if doc1_score <= doc2_score:
                has_issues += 1
                issues.append(f"Issue with {query_id}: doc1 ({doc1_score:.4f}) should be preferred over doc2 ({doc2_score:.4f})")
        
        # For q2, doc2 should be preferred (have higher score)
        if query_id.endswith('_q2'):
            if doc2_score <= doc1_score:
                has_issues += 1
                issues.append(f"Issue with {query_id}: doc2 ({doc2_score:.4f}) should be preferred over doc1 ({doc1_score:.4f})")
    
    print(f"Found {has_issues} issues out of {len(data)} queries")
    return issues

with open("/home/oweller2/my_scratch/mteb/results/home/oweller2/my_scratch/LLaMA-Factory/saves/qwen-7b-v2/lora/sft/checkpoint-4000-full/NevIR_default_predictions.json", "r") as f:
    data = json.load(f)

# Check preferences
issues = check_preferences(data)

# # Print results
# if not issues:
#     print("All preferences are correct!")
# else:
#     print(f"Found {len(issues)} issues:")
#     for issue in issues:
#         print(issue)


"""
ssue with 943-2_q1: doc1 (0.9999) should be preferred over doc2 (1.0000)
Issue with 944-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 944-3_q2: doc2 (0.0003) should be preferred over doc1 (0.9984)
Issue with 948-3_q2: doc2 (0.0000) should be preferred over doc1 (0.9972)
Issue with 949-2_q1: doc1 (0.9999) should be preferred over doc2 (0.9999)
Issue with 951-3_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 952-2_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 952-3_q2: doc2 (0.0004) should be preferred over doc1 (0.0006)
Issue with 954-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 955-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 956-3_q2: doc2 (0.0016) should be preferred over doc1 (0.9949)
Issue with 957-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 957-2_q2: doc2 (0.9999) should be preferred over doc1 (1.0000)
Issue with 958-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 960-2_q2: doc2 (0.9998) should be preferred over doc1 (1.0000)
Issue with 964-3_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 964-3_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 967-2_q2: doc2 (0.9999) should be preferred over doc1 (1.0000)
Issue with 967-3_q2: doc2 (0.9987) should be preferred over doc1 (1.0000)
Issue with 970-2_q2: doc2 (0.0003) should be preferred over doc1 (0.0064)
Issue with 973-3_q2: doc2 (0.0005) should be preferred over doc1 (1.0000)
Issue with 974-3_q2: doc2 (0.0000) should be preferred over doc1 (0.0001)
Issue with 975-2_q2: doc2 (0.9987) should be preferred over doc1 (0.9999)
Issue with 978-2_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 979-3_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 981-2_q2: doc2 (0.9997) should be preferred over doc1 (1.0000)
Issue with 981-3_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 982-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 982-3_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 983-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 984-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 984-2_q2: doc2 (0.0567) should be preferred over doc1 (0.9999)
Issue with 985-3_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 987-3_q2: doc2 (0.9999) should be preferred over doc1 (1.0000)
Issue with 990-2_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 991-2_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 991-3_q1: doc1 (1.0000) should be preferred over doc2 (1.0000)
Issue with 995-2_q1: doc1 (0.0045) should be preferred over doc2 (1.0000)
Issue with 995-2_q2: doc2 (1.0000) should be preferred over doc1 (1.0000)
Issue with 998-3_q1: doc1 (0.9999) should be preferred over doc2 (1.0000)
Issue with 998-3_q2: doc2 (0.9999) should be preferred over doc1 (1.0000)
Issue with 999-2_q1: doc1 (0.9998) should be preferred over doc2 (0.9999)"""