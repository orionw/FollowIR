from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import CohereEmbeddings	
import pandas as pd
import numpy as np
import argparse
import tqdm

from mteb import MTEB
# from sentence_transformers import SentenceTransformer
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel



class APISentenceTransformer(DRESModel):

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.embedding_type = model
        self.embedder = self.load_embedding_model()


    def load_embedding_model(self):
        if self.embedding_type == "openai":
            underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        else:
            underlying_embeddings = CohereEmbeddings(model='embed-english-v3.0')


        self.store = LocalFileStore(f"./embeddings_cache_{self.embedding_type}/")
        print(f"Storing at ./embeddings_cache_{self.embedding_type}/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, self.store, namespace=underlying_embeddings.model
        )
        return cached_embedder


    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, **kwargs)


    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts, **kwargs)

    
    def embed_api(self, sentences, type):
        if type == "docs":
            try:
                return self.embedder.embed_documents(sentences)
            except Exception as e:
                import time
                time.sleep(60)
                return self.embedder.embed_documents(sentences)
        else:
            assert len(sentences) == 1
            sentences = sentences[0]
            try:
                return [self.embedder.embed_query(sentences)]
            except Exception as e:
                import time
                time.sleep(60)
                return [self.embedder.embed_query(sentences)]
        
    

    def encode(self, sentences, batch_size: int = 100, **kwargs):
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
            sentences = [(s + " " + i).strip() for s, i in zip(sentences, instruction_list)]
            print(sentences[0])
            assert len(sentences) == 1
            embeddings = self.embed_api(sentences, type="queries")
            print(len(list(self.store.yield_keys())))

        else:
            # is docs
            embeddings = []
            print(sentences[0], len(sentences))
            iterations = list(range(0, len(sentences), batch_size))
            print(iterations)
            for i in tqdm.tqdm(iterations):
                batch = sentences[i:i+batch_size]
                cur_embeds = self.embed_api(batch, type="docs")
                embeddings.extend(cur_embeds)
            assert len(embeddings) == len(sentences), f"Expected {len(sentences)} embeddings, got {len(embeddings)}."
            print(len(list(self.store.yield_keys())))
            
        return np.array(embeddings)



if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    
    model = APISentenceTransformer(args.model_name_or_path)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, save_corpus_embeddings=True, do_length_ablation=True, batch_size=50)