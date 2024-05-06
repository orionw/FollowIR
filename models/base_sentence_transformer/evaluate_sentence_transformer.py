"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import argparse

from mteb import MTEB
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")



class InstructionSentenceTransformer(SentenceTransformer):

    def __init__(self, is_tart, **kwargs):
        super().__init__(**kwargs)
        self.is_tart = is_tart
    
    def encode_queries(self, queries, batch_size: int, **kwargs):
        if not self.is_tart:
            # return normal, the adding happens already in the model
            return super().encode_queries(queries, batch_size, **kwargs)
        else:
            # TART requires special processing of the sep token
            if self.use_sbert_model:
                if isinstance(self.model._first_module(), Transformer):
                    logger.info(
                        f"Queries will be truncated to {self.model.get_max_seq_length()} tokens."
                    )
                elif isinstance(self.model._first_module(), WordEmbeddings):
                    logger.warning(
                        "Queries will not be truncated. This could lead to memory issues. In that case please lower the batch_size."
                    )

            if "instructions" in kwargs and kwargs["instructions"] is not None:
                queries = [(query + " [SEP] " + kwargs["instructions"][query]).strip() for query in queries]
                new_kwargs = {
                    k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
                }
            else:
                # can't just delete, cuz assign by reference on kwargs
                # and TART always needs an instruction
                queries = [(query + " [SEP] Retrieve news paper paragraph to answer this question").strip() for query in queries]
                new_kwargs = kwargs

            return self.model.encode(queries, batch_size=batch_size, **new_kwargs)
        


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    
    if "tart" in args.model_name_or_path:
        logger.info("Using TART model")
        model = InstructionSentenceTransformer(True, model_name_or_path=args.model_name_or_path)
    else:
        model = SentenceTransformer(model_name_or_path=args.model_name_or_path) # no need for extending it

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"], do_length_ablation=True)  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, save_corpus_embeddings=True,  do_length_ablation=True)