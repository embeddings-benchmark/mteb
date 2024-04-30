## Adding a Model to the MTEB Leaderboard

The MTEB Leaderboard is available [here](https://huggingface.co/spaces/mteb/leaderboard). To submit:

1. Run on MTEB: You can reference [scripts/run_mteb_english.py](https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_english.py) for all MTEB English datasets used in the main ranking, or [scripts/run_mteb_chinese.py](https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_chinese.py) for the Chinese ones. 
Advanced scripts with different models are available in the [mteb/mtebscripts repo](https://github.com/embeddings-benchmark/mtebscripts).
2. Format the json files into metadata using the script at `scripts/mteb_meta.py`. For example
`python scripts/mteb_meta.py path_to_results_folder`, which will create a `mteb_metadata.md` file. If you ran CQADupstack retrieval, make sure to merge the results first with `python scripts/merge_cqadupstack.py path_to_results_folder`.
3. Copy the content of the `mteb_metadata.md` file to the top of a `README.md` file of your model on the Hub. See [here](https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit/blob/main/README.md) for an example.
4. Hit the Refresh button at the bottom of the leaderboard and you should see your scores 🥇
5. To have the scores appear without refreshing, you can open an issue on the [Community Tab of the LB](https://huggingface.co/spaces/mteb/leaderboard/discussions) and someone will restart the space to cache your average scores. The cache is updated anyways ~1x/week.