
# runs all baseline models on a subset of tasks (this is to check tasks are running correctly)

# make sure you install the latest version of mteb:
    # make install
# install codecarbon:
    # pip install codecarbon
# ensure latest version of sentnece-transformers is installed:
    # pip install sentence-transformers --upgrade
# ensure that the the huggingface token is set and accecible using:
    # huggingface-cli login

echo "Running model on a sample set of tasks" # this is to check tasks are running correctly

models=("sentence-transformers/all-MiniLM-L6-v2")
results_folder="results"

for model in "${models[@]}"
do
    echo "Running model: $model"
    mteb run \
    -m $model \
    -t MindSmallReranking SemRel24STS AJGT SummEval NusaTranslationBitextMining \
    --output_folder $results_folder \
    --co2_tracker true
done


echo "Running a sample set of tasks completed successfully!"


echo "Running a sample set of models on one task" # this is to check models implementation are running correctly


models=("sentence-transformers/all-MiniLM-L6-v2" "intfloat/multilingual-e5-small" "intfloat/multilingual-e5-large-instruct")


for model in "${models[@]}"
do
    echo "Running model: $model"
    mteb run \
    -m $model \
    -t LccSentimentClassification \
    --output_folder $results_folder \
    --co2_tracker true
done
