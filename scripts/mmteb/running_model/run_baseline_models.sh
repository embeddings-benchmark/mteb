
# runs all baseline models on all tasks except atm. Retrieval, Clustering, and InstructionRetrieval

models=("sentence-transformers/all-MiniLM-L6-v2" "sentence-transformers/all-MiniLM-L12-v2" "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/LaBSE")
results_folder="{project_root}/results"

for model in "${models[@]}"
do
    echo "Running model: $model"
    mteb run \
    -m $model \
    --task_types BitextMining Classification MultilabelClassification PairClassification Reranking STS Summarization \
    --output_folder $results_folder \
    --co2_tracker
done