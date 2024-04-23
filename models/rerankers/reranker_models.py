from modeling_enc_t5 import EncT5ForSequenceClassification
from tokenization_enc_t5 import EncT5Tokenizer
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import pandas as pd
from math import ceil, exp
from typing import List
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
)
from mteb.evaluation.evaluators.InstructionRetrievalEvaluator import Reranker as MTEB_Reranker

# Based on https://github.com/castorini/pygaggle/blob/f54ae53d6183c1b66444fa5a0542301e0d1090f5/pygaggle/rerank/base.py#L63
prediction_tokens = {
    "castorini/monot5-small-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-small-msmarco-100k": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco": ["▁false", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
}



def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]



class Reranker(MTEB_Reranker):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        fp_options: bool = None,
        silent: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.fp_options = fp_options if fp_options is not None else torch.float32
        if self.fp_options == "auto":
            self.fp_options = torch.float32
        elif self.fp_options == "float16":
            self.fp_options = torch.float16
        elif self.fp_options == "float32":
            self.fp_options = torch.float32
        elif self.fp_options == "bfloat16":
            self.fp_options = torch.bfloat16
        print(f"Using fp_options of {self.fp_options}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.silent = silent
        self.first_print = True

    def rerank(self, query, passages, **kwargs) -> list:
        pass


class MonoT5Reranker(Reranker):
    name: str = "MonoT5"
    prompt_template: str = "Query: {query} Document: {text} Relevant:"

    def __init__(
        self,
        model_name_or_path="castorini/monot5-base-msmarco-10k",
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, **model_args
        )
        print(f"Using model {model_name_or_path}")
        
        if 'torch_compile' in kwargs and kwargs['torch_compile']:
            self.torch_compile = kwargs["torch_compile"]
            self.model = torch.compile(self.model)
        else:
            self.torch_compile = False

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            model_name_or_path,
            self.tokenizer,
            kwargs['token_false'] if 'token_false' in kwargs else None,
            kwargs['token_true'] if 'token_true' in kwargs else None,
        )
        print(f"Using max_length of {self.tokenizer.model_max_length}")
        print(f"Using token_false_id of {self.token_false_id}")
        print(f"Using token_true_id of {self.token_true_id}")
        self.max_length = self.tokenizer.model_max_length
        print(f"Using max_length of {self.max_length}")


        self.model.eval()


    def get_prediction_tokens(self, model_name_or_path, tokenizer, token_false=None, token_true=None):
        if not (token_false and token_true):
            if model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[model_name_or_path]
                token_false_id = tokenizer.get_vocab()[token_false]
                token_true_id  = tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                # raise Exception(f"We don't know the indexes for the non-relevant/relevant tokens for\
                #         the checkpoint {model_name_or_path} and you did not provide any.")
                return self.get_prediction_tokens('castorini/monot5-base-msmarco', self.tokenizer)
        else:
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id


    @torch.inference_mode()
    def rerank(self, queries, passages, **kwargs):
        assert "instructions" in kwargs
        instructions = kwargs["instructions"]
        if instructions is not None and instructions[0] is not None:
            # print(f"Adding instructions to monot5 queries")
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]

        prompts = [
            self.prompt_template.format(query=query, text=text)
            for (query, text) in zip(queries, passages)
        ]
        if self.first_print:
            print(f"Using {prompts[0]}")
            self.first_print = False

        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            pad_to_multiple_of=(8 if self.torch_compile else None),
        ).to(self.device)
        output = self.model.generate(
            **tokens,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        batch_scores = output.scores[0]
        batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()


class LlamaReranker(Reranker):
    name: str = "LLAMA-Based"

    def __init__(
        self, model_name_or_path: str, is_classification: bool = False, **kwargs
    ):
        if "torch_compile" in kwargs:
            del kwargs["torch_compile"]
        super().__init__(model_name_or_path, **kwargs)

        if "chat" in model_name_or_path:
            self.template = LLAMA_CHAT_TEMPLATE = """<s>[INST] <<SYS>>
You are an expert at finding information. Determine if the following document is relevant to the query (true/false).
<</SYS>>Query: {query}
Document: {text}
Relevant: [/INST]"""
        else:
            self.template = """Determine if the following document is relevant to the query (true/false).

Query: {query}
Document: {text}
Relevant: """


        self.query_instruct_template = "{query} {instruction}"
        print(f"Using query_instruct_template of {self.query_instruct_template}")
        self.is_classification = is_classification

        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options

        print(self.template)
        print(model_name_or_path)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_args
        )
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.token_false_id = self.tokenizer.get_vocab()["false"]
        self.token_true_id = self.tokenizer.get_vocab()["true"]
        self.max_length = min(2048, self.tokenizer.model_max_length)
        print(f"Using max_length of {self.max_length}")
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            print(f'Using {self.gpu_count} GPUs')
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()


    @torch.inference_mode()
    def rerank(self, queries, passages, **kwargs):
        assert "instructions" in kwargs
        instructions = kwargs["instructions"]
        if instructions is not None and instructions[0] is not None:
            # print(f"Adding instructions to LLAMA queries")
            queries = [self.query_instruct_template.format(instruction=i, query=q).strip() for i, q in zip(instructions, queries)]

        prompts = [
            self.template.format(query=query, text=text) for (query, text) in zip(queries, passages)
        ]
        assert "{query}" not in prompts[0], "Query not replaced"
        assert "{text}" not in prompts[0], "Text not replaced"
        assert "{instruction}" not in prompts[0], "Instruction not replaced"

        if self.first_print:
            print(f"Using {prompts[0]}")
            self.first_print = False


        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            pad_to_multiple_of=None,
        ).to(self.device)
        if "token_type_ids" in tokens:
            del tokens["token_type_ids"]
        if not self.is_classification:
            batch_scores = self.model(**tokens).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
        else:
            batch_scores = self.model(**tokens).logits
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

        return scores


class MistralReranker(LlamaReranker):
    name: str = "Mistral"

    def __init__(self, model_name_or_path: str, **kwargs):
        # use the base class for everything except template
        super().__init__(model_name_or_path, **kwargs)
        self.template = """<s>[INST] You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false).
Query: {query}
Document: {text}
Relevant (either "true" or "false"): [/INST]"""
        self.max_length = min(2048, self.tokenizer.model_max_length)
        print(f"Using max_length of {self.max_length}")
        print(f"Using template of {self.template}")


class GritLMReranker(MistralReranker):
    name: str = "GritLM"

    def __init__(self, model_name_or_path: str, **kwargs):
        model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
        super().__init__(model_name_or_path, **kwargs)
        from gritlm import GritLM
        grit = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
        self.template = "<|user|>\nRank the passage based on its relevance to the search query (true/false) {query}.\n\n{text}\n\n" \
                   "Search Query: {query}.\n\n" \
                   "Determine the passage's relevance to the search query." \
                   "Only respond with true or false, do not say any other word or explain.\n<|assistant|>\n"
        self.model = grit.model
        self.tokenizer = grit.tokenizer


class FollowIRReranker(LlamaReranker):
    name: str = "FollowIR"

    def __init__(self, model_name_or_path: str, **kwargs):
        # use the base class for everything except template
        super().__init__(model_name_or_path, **kwargs)
        self.template = """<s> [INST] You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false). Answer using only one word, one of those two choices.

Query: {query}
Document: {text}
Relevant (only output one word, either "true" or "false"): [/INST] """
        self.max_length = min(2048, self.tokenizer.model_max_length)
        # self.query_instruct_template = "\"{query}\", details: \"{instruction}\""
        print(f"Using query_instruct_template of {self.query_instruct_template}")
        print(f"Using template of {self.template}")



class FLANT5Reranker(MonoT5Reranker):
    name: str = "FLAN-T5"
    prompt_template: str = """Is the following passage relevant to the query?
Query: {query}
Passage: {text}"""

    def get_prediction_tokens(self, *args, **kwargs):
        yes_token_id, *_ = self.tokenizer.encode("yes")
        no_token_id, *_ = self.tokenizer.encode("no")
        return no_token_id, yes_token_id



class MonoBERTReranker(Reranker):
    name: str = "MonoBERT"

    def __init__(
        self,
        model_name_or_path="castorini/monobert-large-msmarco",
        torch_compile=False,
        **kwargs,
    ):
        super().__init__(model_name_or_path, **kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp_options:
            model_args["torch_dtype"] = self.fp_options
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **model_args,
        )
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = self.tokenizer.model_max_length
        print(f"Using max_length of {self.max_length}")

        self.model.eval()


    @torch.inference_mode()
    def rerank(self, queries, passages, **kwargs):
        assert "instructions" in kwargs
        instructions = kwargs["instructions"]
        if instructions is not None and instructions[0] is not None:
            # print(f"Adding instructions to MonoBERT queries")
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]


        if self.first_print:
            print(f"Using {queries[0]}")
            self.first_print = False

        tokens = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation="only_second",
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        output = self.model(**tokens)[0]
        batch_scores = torch.nn.functional.log_softmax(output, dim=1)
        return batch_scores[:, 1].exp().tolist()


class TARTFullReranker(Reranker):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        assert model_name_or_path in ["facebook/tart-full-flan-t5-xl", "facebook/tart-full-t0-3b"]
        self.model = EncT5ForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer =  EncT5Tokenizer.from_pretrained(model_name_or_path)
        self.max_length = 1024
        print(f"Using max_length of {self.max_length}")

        self.model.eval()


    @torch.inference_mode()
    def rerank(self, queries, passages, **kwargs):
        assert "instructions" in kwargs
        instructions = kwargs["instructions"]
        if instructions is not None and instructions[0] is not None:
            # combine them with the queries with a [SEP] token
            if instructions[0].strip() == "": # empty instruction case, use generic
                queries = [f"{query} [SEP] Retrieve news paper paragraph to answer this question" for query in queries]
            else:
                queries = [f"{query} [SEP] {instruction}".strip() for query, instruction in zip(queries, instructions)]

        assert len(queries) == len(
            passages
        ), "queries and passages must be the same length"
        for query in queries:
            assert " [SEP] " in query, "query must contain [SEP]"

        if torch.cuda.is_available():
            self.model.to("cuda")
            # print("Loaded model to cuda")

        if self.first_print:
            print(f"Using {queries[0]}")
            self.first_print = False

        self.model.eval()

        features = self.tokenizer(
            queries,
            passages,
            padding=True,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
        )
        
        if torch.cuda.is_available():
            features = {k: v.to("cuda") for k, v in features.items()}
        with torch.no_grad():
            scores = self.model(**features).logits
            normalized_scores = [
                float(score[1]) for score in F.softmax(scores, dim=1)
            ]
        return normalized_scores



class RankLlamaReranker(Reranker):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

        from peft import PeftModel, PeftConfig
        def get_model(peft_model_name):
            config = PeftConfig.from_pretrained(peft_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
            model = PeftModel.from_pretrained(base_model, peft_model_name)
            model = model.merge_and_unload()
            model.eval()
            return model

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        self.model = get_model('castorini/rankllama-v1-7b-lora-passage')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()


    @torch.inference_mode()
    def rerank(self, queries, passages, **kwargs):
        assert "instructions" in kwargs
        instructions = kwargs["instructions"]
        if instructions is not None and instructions[0] is not None:
            # print(f"Adding instructions to RankLlama queries")
            queries = [f"{q} {i}".strip() for i, q in zip(instructions, queries)]
        assert len(queries) == len(
            passages
        ), "queries and passages must be the same length"

        if torch.cuda.is_available():
            self.model.to("cuda")
            # print("Loaded model to cuda")

        if self.first_print:
            print(f"Using {queries[0]}")
            self.first_print = False

        self.model.eval()
       
        cur_queries = [f'query: {query}' for query in queries]
        cur_passages = [f'document: {passage}' for passage in passages]
        features = self.tokenizer(
            cur_queries,
            cur_passages,
            padding=True,
            return_tensors="pt",
            truncation="only_second",
            max_length=1024,
        )

        if torch.cuda.is_available():
            features = {k: v.to("cuda") for k, v in features.items()}
        with torch.no_grad():
            scores = self.model(**features).logits

        return scores.cpu().tolist()



MODEL_DICT = {
    "facebook/tart-full-flan-t5-xl": TARTFullReranker,
    "castorini/monot5-small-msmarco-10k": MonoT5Reranker,
    "castorini/monot5-base-msmarco-10k": MonoT5Reranker,
    "castorini/monot5-large-msmarco-10k": MonoT5Reranker,
    "castorini/monot5-3b-msmarco-10k": MonoT5Reranker,
    "google/flan-t5-base": FLANT5Reranker,
    "google/flan-t5-large": FLANT5Reranker,
    "google/flan-t5-xl": FLANT5Reranker,
    "google/flan-t5-xxl": FLANT5Reranker,
    "castorini/monobert-large-msmarco": MonoBERTReranker,
    "meta-llama/Llama-2-7b-hf": LlamaReranker,
    "meta-llama/Llama-2-7b-chat-hf": LlamaReranker,
    "mistralai/Mistral-7B-Instruct-v0.2": MistralReranker,
    # "castorini/rankllama-v1-7b-lora-passage": RankLlamaReranker, # Not working correctly
    "custom_mistral": FollowIRReranker,
    "GritLM": GritLMReranker,
}
