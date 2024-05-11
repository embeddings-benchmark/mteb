## Adding a new Leaderboard tab

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard) and we love new leaderboard tabs. To add a new leaderboard tab:

1. Open a PR in https://hf.co/datasets/mteb/results with:
- All results added in existing model folders or new folders
- Updated paths.json (see snippet results.py)
- If adding any new models, their names added to results.py
- If you have access to all models you are adding, you can also [add results via the metadata](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) for all of them / some of them
2. Open a PR at https://huggingface.co/spaces/mteb/leaderboard modifying app.py to add your tab:
- Add any new models & their specs to the global lists
- Add your tab, credits etc to where the other tabs are defined
- If you're adding new results to existing models, remove those models from `EXTERNAL_MODEL_RESULTS.json` such that they can be reloaded with the new results and are not cached.
- You may also have to uncomment `, download_mode='force_redownload', verification_mode="no_checks")` where the datasets are loaded to experiment locally without caching of results
- Test that it runs & works locally as you desire with python app.py, **please add screenshots to the PR**
