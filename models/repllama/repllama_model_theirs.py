"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from mteb import MTEB
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings

from transformers import AutoTokenizer, AutoModel
from peft import PeftConfig, PeftModel
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel, is_dres_compatible

# from repllama import RepLLaMA
from utils import pool, move_to_cuda, create_batch_dict

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")



class RepLlamaModel(DRESModel):

    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model=None, **kwargs)
        self.base_model = "meta-llama/Llama-2-7b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model
        )
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "right"

        self.model = self.get_model(model_name_or_path).cuda()

    def get_model(self, peft_model_name):
        base_model = AutoModel.from_pretrained(self.base_model)
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        model.eval()
        return model

    def encode_queries(self, queries, batch_size: int, **kwargs):
        if "instructions" in kwargs and kwargs["instructions"] is not None:
            queries = [("query: " + query + " " + kwargs["instructions"][query]).strip() + "</s>" for query in queries]
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            assert False
            queries = [("Query: " + query).strip() for query in queries]
            new_kwargs = kwargs

        return self._encode(queries, batch_size=batch_size, **new_kwargs)

    def encode_corpus(self, passages, batch_size: int, **kwargs):
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() + "</s>" for doc in passages]
        passages = [("passage: " + passage).strip() for passage in input_texts]
        return self._encode(passages, batch_size=batch_size, **kwargs)
    

    @torch.no_grad()
    def _encode(self, input_texts, batch_size, **kwargs) -> np.ndarray:
        encoded_embeds = []
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts = input_texts[start_idx: start_idx + batch_size]
            batch_dict = self.tokenizer(batch_input_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

            batch_dict = move_to_cuda(batch_dict)

            with torch.no_grad():
                # compute query embedding
                query_outputs = self.model(**batch_dict)
                query_embedding = query_outputs.last_hidden_state[:, -1]
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
                encoded_embeds.extend(query_embedding.cpu().numpy())

        assert len(encoded_embeds) == len(input_texts), f"{len(encoded_embeds)} != {len(input_texts)}"
        dim = encoded_embeds[0].shape[0]
        to_ret = np.concatenate(encoded_embeds, axis=0).reshape(-1, dim)
        print(to_ret.shape)
        return to_ret


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    

    model = RepLlamaModel(model_name_or_path=args.model_name_or_path)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"], do_length_ablation=True)  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, save_corpus_embeddings=True, batch_size=4, do_length_ablation=True)


    # python -u models/repllama/repllama_model.py --model_name_or_path "castorini/repllama-v1-7b-lora-passage" --output_dir "results/castorini--repllama-v1-7b-lora-passage" 

    # python -u models/repllama/repllama_model.py --model_name_or_path "/home/hltcoe/oweller/my_exps/tevatron/retriever-llama2-instruct/checkpoint-200/" --output_dir "results/retriever-llama2-instruct"  

    # python -u models/repllama/repllama_model.py --model_name_or_path "/home/hltcoe/oweller/my_exps/tevatron/retriever-llama2/checkpoint-200/" --output_dir "results/retriever-llama2"  
        
        