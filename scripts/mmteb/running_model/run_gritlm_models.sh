
# runs all baseline models on all tasks except atm. Retrieval, Clustering, and InstructionRetrieval

models=("GritLM/GritLM-7B" "GritLM/GritLM-8x7B")
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