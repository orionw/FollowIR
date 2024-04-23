
import logging
import argparse

from mteb import MTEB

logging.basicConfig(level=logging.INFO)
from reranker_models import * 


logger = logging.getLogger("main")


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--fp_options", default="bfloat16", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    print(args)
    
    if args.model_name_or_path in MODEL_DICT:
        model = MODEL_DICT[args.model_name_or_path](args.model_name_or_path, fp_options=args.fp_options)
    else:
        model = MODEL_DICT["custom_mistral"](args.model_name_or_path, fp_options=args.fp_options)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval'], task_langs=['eng']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, batch_size=args.batch_size)


#  python models/rerankers/evaluate_reranker.py --model_name_or_path castorini/monot5-base-msmarco-10k --output_dir results/castorini--monot5-base-msmarco-10k --batch_size 128 --fp_options float16