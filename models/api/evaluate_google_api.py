import pandas as pd
import numpy as np
import argparse
import tqdm
import time

from mteb import MTEB
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel

from typing import List

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel



class APISentenceTransformerGoogle(DRESModel):

    def __init__(self, model, **kwargs):
        super().__init__(model="text-embedding-preview-0409", **kwargs)
        self.embedder = TextEmbeddingModel.from_pretrained(model)
        self.model_name = model

    def embed_text(
        self,
        texts: List[str],
        task_embed: str,
    ) -> List[List[float]]:
        """Embeds texts with a pre-trained, foundational model."""
        inputs = [TextEmbeddingInput(text, task_embed) for text in texts]
        embeddings = self.embedder.get_embeddings(inputs)
        return [embedding.values for embedding in embeddings]


    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, **kwargs)


    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts, **kwargs)
   

    def encode(self, sentences, batch_size: int = 20, **kwargs):
        """
        Computes sentence embeddings from an API

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
  
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        if "instructions" in kwargs: # is queries
            instructions = kwargs["instructions"]
            instruction_list = [instructions[q].strip() for q in sentences]
            sentences = [(i + " " + s).strip() for s, i in zip(sentences, instruction_list)]
            print(sentences[0])
            assert len(sentences) == 1
            try:
                embeddings = self.embed_text(sentences, "RETRIEVAL_QUERY")
            except Exception as e:
                print(sentences)
                time.sleep(30)
                return self.encode(sentences, batch_size=1, **kwargs)

        else:
            # is docs
            batch_size = 10
            embeddings = []
            print(sentences[0], len(sentences))
            iterations = list(range(0, len(sentences), batch_size))
            print(iterations)
            for i in tqdm.tqdm(iterations):
                batch = sentences[i:i+batch_size]
                try:
                    cur_embeds = self.embed_text(batch, "RETRIEVAL_DOCUMENT")   
                except Exception as e:
                    print(batch)
                    time.sleep(30)
                    return self.encode(sentences, batch_size=1, **kwargs)
                embeddings.extend(cur_embeds)
            assert len(embeddings) == len(sentences), f"Expected {len(sentences)} embeddings, got {len(embeddings)}."
            
        return np.array(embeddings)



if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    
    model = APISentenceTransformerGoogle(args.model_name_or_path)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"], do_length_ablation=True)  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, save_corpus_embeddings=True, batch_size=50)