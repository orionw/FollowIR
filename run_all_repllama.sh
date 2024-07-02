#!/bin/bash

python -u models/repllama/repllama_model.py --model_name_or_path "/home/hltcoe/oweller/my_exps/tevatron/retriever-llama2/checkpoint-400/" --output_dir "results/retriever-llama2-400" 

python -u models/repllama/repllama_model.py --model_name_or_path "/home/hltcoe/oweller/my_exps/tevatron/retriever-llama2-instruct/checkpoint-400/" --output_dir "results/retriever-llama2-instruct-400"  

python -u models/repllama/repllama_model.py --model_name_or_path "/home/hltcoe/oweller/my_exps/tevatron/retriever-llama2/checkpoint-600/" --output_dir "results/retriever-llama2-600" 

python -u models/repllama/repllama_model.py --model_name_or_path "/home/hltcoe/oweller/my_exps/tevatron/retriever-llama2-instruct/checkpoint-600/" --output_dir "results/retriever-llama2-instruct-600"  
