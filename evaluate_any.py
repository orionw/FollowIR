import os
import argparse
import json


def evaluate_any(args):
    """
    This script takes in the arguments and creates the arguments for the bash file with the requirements given from any of the models

    Args:
    args: the arguments from the command line

    Returns:
    a string of the command to be used
    """
    args.output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "__"))
    
    if args.model_name in ["BAAI/bge-base", "BAAI/bge-large-en", "BAAI/bge-small-en", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", "BAAI/bge-small-en-v1.5"]:
        cmd = f"python -u models/bge/evaluate_bge.py --model_name_or_path {args.model_name} --output_dir {args.output_dir}"
    

    elif args.model_name in ["openai", "cohere", "google"]:
        return f"python models/api/evaluate_api.py --model_name_or_path {args.model_name} --output_dir {args.output_dir}"


    elif args.model_name in ["bm25"]:
        return f"python -u models/bm25/evaluate_bm25.py --output_dir {args.output_dir}"
    

    elif args.model_name in ["intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2", "intfloat/e5-small", "intfloat/e5-base", "intfloat/e5-large", "intfloat/e5-mistral-7b-instruct"]:
        if args.model_name == "infloat/e5-mistral-7b-instruct":
            pool_type = "last"  # pool_type should be last
            batch_size = args.batch_size
        else:
            pool_type = "avg"
            batch_size = 128
            
        cmd = f"python -u models/e5/evaluate_e5.py --model_name_or_path {args.model_name} --output_dir {args.output_dir}  --pool_type {pool_type} --batch_size {batch_size}"


    elif args.model_name in ["hkunlp/instructor-xl", "hkunlp/instructor-large", "hkunlp/instructor-base"]:
        cmd = f"python -u models/instructor/evaluate_instructor.py --model_name {args.model_name} --output_dir {args.output_dir}"


    elif args.model_name in ["castorini/monobert-large-msmarco", "facebook/tart-full-flan-t5-xl", "castorini/monot5-small-msmarco-10k", "castorini/monot5-base-msmarco-10k", "castorini/monot5-large-msmarco-10k", "castorini/monot5-3b-msmarco-10k", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-3b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "GritLM"] or "custom_mistral" in args.model_name:

        if "custom_mistral" in args.model_name:
            args.model_name = args.model_name.replace("custom_mistral--", ""
                                                      )
        cmd = f"python -u models/rerankers/evaluate_reranker.py --model_name_or_path {args.model_name} --output_dir {args.output_dir} --fp_options {args.fp_options} --batch_size {args.batch_size}"
    

    elif args.model_name in ["GritLM/GritLM-7B"]:
        cmd = f"python -u models/gritlm/evaluate_gritlm.py --model_name_or_path {args.model_name} --output_dir {args.output_dir}"


    else:
        if 'tart-dual-contriever' in args.model_name:
            args.model_name = "orionweller/tart-dual-contriever-msmarco-copy"

        cmd = f"python -u models/base_sentence_transformer/evaluate_sentence_transformer.py --model_name_or_path {args.model_name} --output_dir {args.output_dir}"


    if args.task_names:
        cmd += f" --task_names {' '.join(args.task_names)}"

    return cmd



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--fp_options", default="bfloat16", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    cmd = evaluate_any(args)
    print(cmd)