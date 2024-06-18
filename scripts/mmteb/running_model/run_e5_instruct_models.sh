
# runs all baseline models on all tasks except atm. Retrieval, Clustering, and InstructionRetrieval

models=("intfloat/multilingual-e5-large-instruct" "intfloat/e5-mistral-7b-instruct")
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