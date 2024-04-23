"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import argparse

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")



class InstructionSentenceTransformer(SentenceTransformer):

    def __init__(self, is_tart, **kwargs):
        super().__init__(**kwargs)
        self.is_tart = is_tart
    

    def encode(self, sentences, 
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False,
               **kwargs):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        # NOTE: this is handled for all models except TART in MTEB's new code, but leaving this in case it's easier for others to modify
        if "instructions" in kwargs: # is queries
            instructions = kwargs["instructions"]
            instruction_list = [instructions[q].strip() for q in sentences]
            if self.is_tart:
                # if instruction_list has empty, use a generic
                instruction_list = [i if i.strip() != "" else "Retrieve news paper paragraph to answer this question" for i in instruction_list]
            self.sep = " " if not self.is_tart else " [SEP] "
            sentences = [(s + self.sep + i).strip() for s, i in zip(sentences, instruction_list)]
            print(sentences[0])
        
        return super().encode(sentences, batch_size, show_progress_bar, output_value, convert_to_numpy, convert_to_tensor, device, normalize_embeddings)


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    
    model = InstructionSentenceTransformer("tart" in args.model_name_or_path, model_name_or_path=args.model_name_or_path)

    if args.task_names is None:
        task_names = [t.description["name"] for t in MTEB(task_types=['InstructionRetrieval'], task_langs=['en']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, save_corpus_embeddings=True)