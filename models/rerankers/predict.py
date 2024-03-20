import json
import os
import argparse
import pandas as pd
import tqdm

import reranker_models as models


def predict(args):
    # read in the `args.data` file as json
    # load the reranker from `reranker_models` and batch over the data

    with open(args.data, "r") as fin:
        data = json.load(fin)

    if args.debug:
        data = data[:100]

    if args.model_name_or_path in models.MODEL_DICT:
        model = models.MODEL_DICT[args.model_name_or_path](args.model_name_or_path, fp_options=args.fp_options)
    else:
        model = models.MODEL_DICT["custom_mistral"](args.model_name_or_path, fp_options=args.fp_options)

    results = []
    # batch over args.batch_size
    for i in tqdm.tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i+args.batch_size]
        queries = [item["query"] for item in batch]
        passages = [item["document"] for item in batch]
        instructions = [item["narrative"] for item in batch]
        labels = [item["output"] for item in batch]
        ids = [item["id"] for item in batch]
        scores = model.rerank(queries, passages, instructions=instructions)
        for i, score in enumerate(scores):
            results.append({
                "score": score,
                "label": labels[i],
                "id": ids[i],
                "query": queries[i],
                "document": passages[i],
                "instruction": instructions[i],
            })

    # write the results to a file
    with open(args.output, "w") as fout:
        json.dump(results, fout, indent=4)
    print(f"Saved to {args.output}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--data", default=None, type=str)
    parser.add_argument("--output", default="results.json", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--fp_options", default="bfloat16", type=str)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    predict(args)

    # example: python models/rerankers/predict.py --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 --data /home/hltcoe/oweller/my_exps/mteb-instruct/training/prompting/outputs_v1/relevance_narrative_separated.json --output results.json --batch_size 32 --fp_options bfloat16 --output mistral-generations.json
    # for TART-FLAN-t5-xl: python models/rerankers/predict.py --model_name_or_path facebook/tart-full-flan-t5-xl --data /home/hltcoe/oweller/my_exps/mteb-instruct/training/prompting/outputs_v1/relevance_narrative_separated.json --output results.json --batch_size 32 --fp_options bfloat16 --output tart-generations.json