#!/bin/bash

# define an array of models
models=("hkunlp/instructor-xl" "hkunlp/instructor-base" "BAAI/bge-base-en-v1.5" "BAAI/bge-large-en-v1.5" "intfloat/e5-base-v2" "intfloat/e5-large-v2"  "intfloat/e5-mistral-7b-instruct" "castorini/monobert-large-msmarco" "facebook/tart-full-flan-t5-xl"  "castorini/monot5-base-msmarco-10k"  "castorini/monot5-3b-msmarco-10k" "google/flan-t5-base" "google/flan-t5-large"  "meta-llama/Llama-2-7b-chat-hf" "mistralai/Mistral-7B-Instruct-v0.2" "GritLM/GritLM-7B" "tart-dual-contriever-msmarco" "facebook/contriever-msmarco"  "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-29" "facebook/contriever-msmarco"  "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-232")

# models=("intfloat/e5-base-v2")

# "google/flan-t5-xxl" "meta-llama/Llama-2-7b-hf" "infloat/e5-small" "intfloat/e5-small-v2" "hkunlp/instructor-large"  "BAAI/bge-small-en" "BAAI/bge-base-en" "BAAI/bge-small-en-v1.5" "intfloat/e5-large" "castorini/monot5-small-msmarco-10k" "castorini/monot5-large-msmarco-10k"

models=("intfloat/e5-mistral-7b-instruct")

# models=("hkunlp/instructor-xl"  "hkunlp/instructor-base" "BAAI/bge-large-en" "BAAI/bge-small-en" "BAAI/bge-large-en-v1.5" "BAAI/bge-small-en-v1.5" "intfloat/e5-small-v2" "intfloat/e5-large-v2" "infloat/e5-small" "intfloat/e5-large" "infloat/e5-mistral-7b-instruct" "castorini/monobert-large-msmarco" "facebook/tart-flan-t5-xl" "castorini/monot5-small-msmarco-10k" "castorini/monot5-base-msmarco-10k" "castorini/monot5-large-msmarco-10k" "castorini/monot5-3b-msmarco-10k"  "google/flan-t5-3b" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-7b-chat-hf" "mistralai/Mistral-7B-Instruct-v0.2" "GritLM/GritLM-7B" "tart-dual-contriever-msmarco")

# make a small set of these for testing
# models=("intfloat/e5-small" "castorini/monobert-large-msmarco" "google/flan-t5-base")
# models=("mistralai/Mistral-7B-Instruct-v0.2" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-29" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-58" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-87" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-116" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-145" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-174" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-203" "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v2-232" ) # "custom_mistral--/home/hltcoe/oweller/my_exps/LLaMA-Factory/followir-v1/" 
# models=("google/flan-t5-xxl" "google/flan-t5-xl")

# example job: qsub -N rerank -j y -o "/home/hltcoe/oweller/my_exps/mteb-instruct/logs/$model/rerank.out.$((`date '+%s%N'`/1000))" -l h_rt=96:00:00,gpu=1 -q gpu.q@@a100 launch_jobs.sh --model_name castorini/monot5-3b-msmarco-10k --batch_size 32 

# loop over all models and launch a job for them
for model in "${models[@]}"
do
    file_safe_model_name=$(echo $model | tr / _)
    echo "Launching job for model: $model"
    qsub -N rerank -j y -o "/home/hltcoe/oweller/my_exps/mteb-instruct/logs/$file_safe_model_name/rerank.out.$((`date '+%s%N'`/1000))" -l h_rt=96:00:00,gpu=1 -q gpu.q@@a100 launch_jobs.sh --model_name $model --batch_size 8
done
