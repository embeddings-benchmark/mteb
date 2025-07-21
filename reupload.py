# from __future__ import annotations

# from datasets import load_dataset

# ds = load_dataset(
#     "facebook/voxpopuli",
#     "en_accented",
#     revision="719aaef8225945c0d80b277de6c79aa42ab053d5",
#     trust_remote_code=True,
# )
# # import pdb; pdb.set_trace()
# ds = ds.class_encode_column("accent")

# ds["test"] = ds["test"].train_test_split(
#     test_size=2048, seed=42, stratify_by_column="accent"
# )["test"]

# # import pdb; pdb.set_trace()

# from huggingface_hub import create_repo

# repo = "mteb/voxpopuli-accent-clustering"
# create_repo(repo, repo_type="dataset")
# ds.push_to_hub(repo)
