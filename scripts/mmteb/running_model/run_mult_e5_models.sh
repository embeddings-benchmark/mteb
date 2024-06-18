
# runs all multilingual e5 models on all tasks except atm. retrieval, clustering, and instruction retrieval


# --task_types BitextMining Classification Clustering InstructionRetrieval MultilabelClassification PairClassification Reranking Retrieval STS Summarization

models=("intfloat/multilingual-e5-small" "intfloat/multilingual-e5-base" "intfloat/multilingual-e5-large")
results_folder="results"

for model in "${models[@]}"
do
    echo "Running model: $model"
    mteb run \
    -m $model \
    --task_types BitextMining Classification MultilabelClassification PairClassification Reranking STS Summarization \
    --output_folder $results_folder \
    --co2_tracker true
done