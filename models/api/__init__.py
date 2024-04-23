import argparse

from mteb import MTEB
from rank_bm25 import BM25Okapi
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


class APIModel(MTEB_Reranker):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bm25 = None


    def clean(self, s: str) -> str:
        s = s.lower()
        s = " ".join([stemmer.stem(w) for w in s.split(" ") if w not in stop_words])
        return s

    def rerank(self, queries, passages, **kwargs):
        breakpoint()
        if "instructions" in kwargs: # is queries
            instructions = kwargs["instructions"]
            instruction_list = [self.clean(instructions[q].strip()) for q in instructions]
            query_list = [self.clean(q) for q in queries]
            ready_queries = [(s + " " + i).strip() for s, i in zip(query_list, instruction_list)]
            print(ready_queries[0])
        else:
            lowercase_and_stemmed_and_no_stopwords = [self.clean(item) for item in passages]
            tokenized_corpus = [doc.split(" ") for doc in lowercase_and_stemmed_and_no_stopwords]
            print(tokenized_corpus[0])
            self.bm25 = BM25Okapi(tokenized_corpus)

        scores = []
        for query in ready_queries:
            tokenized_query = query.split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)
            scores.extend(doc_scores)
        
        return scores


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="results", type=str)
    parser.add_argument("--task_names", default=None, type=str, nargs='+')
    args = parser.parse_args()
    
    model = APIModel(args.model_name_or_path)

    if args.task_names is None:
        task_names = [t.metadata_dict["name"] for t in MTEB(task_types=['InstructionRetrieval']).tasks]
    else:
        task_names = args.task_names

    for task in task_names:
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model, output_folder=args.output_dir, eval_splits=eval_splits)