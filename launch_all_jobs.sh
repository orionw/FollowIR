#!/bin/bash

# define an array of models
models=("bm25" "intfloat/e5-base-v2" "intfloat/e5-large-v2" "facebook/contriever-msmarco" "castorini/monobert-large-msmarco" "castorini/monot5-base-msmarco-10k" "castorini/monot5-3b-msmarco-10k" "cohere" "openai" "google" "BAAI/bge-base-en" "BAAI/bge-large-en" "tart-dual-contriever-msmarco" "hkunlp/instructor-base" "hkunlp/instructor-xl" "facebook/tart-full-flan-t5-xl" "intfloat/e5-mistral-7b-instruct"  "GritLM/GritLM-7B" "google/flan-t5-base" "google/flan-t5-large"  "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-7b-chat-hf" "GritLM" "mistralai/Mistral-7B-Instruct-v0.2" "jhu-clsp/FollowIR-7B")

mkdir -p results
mkdir -p logs

# loop over all models and launch a job for them
for model in "${models[@]}"
do
    file_safe_model_name=$(echo $model | tr / _)
    echo "Launching job for model: $model"
    qsub -N rerank -j y -o "/home/hltcoe/oweller/my_exps/FollowIR/logs/$file_safe_model_name/rerank.out.$((`date '+%s%N'`/1000))" -l h_rt=96:00:00,gpu=1 -q gpu.q@@a100 launch_jobs.sh --model_name $model --batch_size 8
done
