"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import argparse

from mteb import MTEB
from sentence_transformers import SentenceTransformer
from llm2vec import LLM2Vec

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel, is_dres_compatible


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


class LLM2Vec_MTEB(DRESModel):

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        # additional is the model_name after -mntp
        pre_model_name = model_name.split("-mntp")[0] + "-mntp"
        tokenizer = AutoTokenizer.from_pretrained(pre_model_name)
        config = AutoConfig.from_pretrained(pre_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            pre_model_name,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = PeftModel.from_pretrained(
            model,
            pre_model_name,
        )
        model = model.merge_and_unload()  # This can take several minutes on cpu

        # Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
        model = PeftModel.from_pretrained(
            model, model_name
        )

        # Wrapper for encoding and pooling operations
        self.l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        self.tokenizer = AutoTokenizer.from_pretrained(pre_model_name)
    
    def encode_queries(self, queries, batch_size: int, **kwargs):
        # Encoding queries using instructions

        # Option 1, use a task prefix also
        # queries = [("Given a web search query, retrieve relevant passages that answer the query: " + query + " " + kwargs["instructions"][query]).strip() for query in queries]

        # Option 2, use the instructions as the task prefix
        queries = [(kwargs["instructions"][query] +  ": " + query).strip() for query in queries]

        q_reps = self.l2v.encode(queries)
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        return q_reps_norm


    def encode_corpus(self, documents, batch_size: int, **kwargs):
        # Encoding documents. Instruction are not required for documents
        text_documents = [doc["title"] + " " + doc["text"] for doc in documents]
        d_reps = self.l2v.encode(text_documents)
        d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
        return d_reps_norm
        

if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    
    model = LLM2Vec_MTEB(args.model_name_or_path) # no need for extending it

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"], do_length_ablation=True)  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, save_corpus_embeddings=True)