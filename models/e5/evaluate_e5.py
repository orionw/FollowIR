import os
import json
import tqdm
import numpy as np
import torch
import argparse
import torch.nn.functional as F

from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel, is_dres_compatible
from utils import pool, logger, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, create_batch_dict
from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE

parser = argparse.ArgumentParser(description='evaluation for BEIR benchmark')
parser.add_argument('--model_name_or_path', default='intfloat/e5-small-v2',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--output_dir', default='tmp-outputs/',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='batch size')
parser.add_argument('--doc-as-query', action='store_true', help='use query prefix for passages, only used for Quora as it is a symmetric task')
parser.add_argument('--pool_type', default='avg', help='pool type')
parser.add_argument('--prefix_type', default='query_or_passage', help='prefix type')
parser.add_argument('--task_names', default=None, type=str, nargs='+', help='task names')

args = parser.parse_args()
base_name: str = args.model_name_or_path.split('/')[-1]
args.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, args.pool_type)
args.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, args.prefix_type)

logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
assert args.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'
os.makedirs(args.output_dir, exist_ok=True)


class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, **kwargs):
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        assert "instructions" in kwargs
        instructions = kwargs["instructions"]
        if args.prefix_type == 'query_or_passage':
            combined_query_and_instruction = [(q + " " + instructions[q]).strip() for q in queries]
            input_texts = [f'query: {q.strip()}' for q in combined_query_and_instruction]
        else:
            instruction_list = [instructions[q] for q in queries]
            generic = "Given a web search query, retrieve relevant passages that answer the query"
            instruction_list = [generic if i == "" else i for i in instruction_list]
            input_texts = [(q + " " + i).strip() for q, i in zip(queries, instruction_list)]

        print(input_texts[0])
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if args.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        if args.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]

        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 64 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, always_add_eos=(args.pool_type == 'last'))
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


def main():
    assert is_dres_compatible(RetrievalModel)
    model = RetrievalModel()

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names
    logger.info('Tasks: {}'.format(task_names))

    for task in task_names:

        logger.info('Processing task: {}'.format(task))

        task_def: str = get_task_def_by_task_name_and_type(task_name=task, task_type='InstructionRetrieval')
        prompt: str = get_detailed_instruct(task_def)
        model.set_prompt(prompt=prompt)
        logger.info('Set prompt: {}'.format(prompt))

        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(model, eval_splits=["test"],
                       output_folder=args.output_dir,
                       save_corpus_embeddings=True, do_length_ablation=True,
                       batch_size=args.batch_size)


if __name__ == '__main__':
    main()