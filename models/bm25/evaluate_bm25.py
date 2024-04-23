import argparse

from mteb import MTEB
from rank_bm25 import BM25Okapi
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


class BM25Reranker(DenseRetrievalExactSearch):

    def __init__(self, **kwargs):
        super().__init__("bm25", **kwargs)
        self.bm25 = None


    def clean(self, s: str) -> str:
        s = s.lower()
        s = " ".join([stemmer.stem(w) for w in s.split(" ") if w not in stop_words])
        # remove punctuation
        s = s.translate(str.maketrans('', '', string.punctuation))
        return s.strip()

    def rerank(self, queries, passages, **kwargs):
        assert len(set(queries)) == 1
        queries = [queries[0]]

        lowercase_and_stemmed_and_no_stopwords = [self.clean(item) for item in passages]
        tokenized_corpus = [[item for item in doc.split(" ") if item] for doc in lowercase_and_stemmed_and_no_stopwords]
        print(tokenized_corpus[0][:10])
        self.bm25 = BM25Okapi(tokenized_corpus)
            
        if "instructions" in kwargs: # is queries
            instructions = kwargs["instructions"]
            assert len(set(instructions)) == 1
            instructions = [instructions[0]]
            instruction_list = [self.clean(i.strip()) for i in instructions]
        else:
            instructions = [""]

        query_list = [self.clean(q) for q in queries]
        ready_queries = [(s + " " + i).strip() for s, i in zip(query_list, instruction_list)]
        assert len(ready_queries) == 1
        # remove any empty 
        print(ready_queries[0][:10])

        tokenized_query = [item for item in ready_queries[0].split(" ") if item]
        doc_scores = self.bm25.get_scores(tokenized_query)
        assert len(doc_scores) == len(passages)
        # pad to length of all scores
        scores = doc_scores.tolist()
        
        assert len(scores) == len(passages)
        return scores


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    
    model = BM25Reranker()

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval'], task_langs=['eng']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits, batch_size=999999)