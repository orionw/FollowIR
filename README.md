<h1 align="center">FollowIR: Evaluating and Teaching Information
Retrieval Models to Follow Instructions</b></h1>

<h4 align="center">
    <p>
        <a href="#links">Links</a> |
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="#leaderboard">Leaderboard (coming soon)</a> |
        <a href="#citing">Citing</a> |
    <p>
</h4>

Official repository for the paper [FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions (https://arxiv.org/abs/2403.15246) Evaluation extends the [MTEB](https://github.com/embeddings-benchmark/mteb) framework to use instructions, so you can evaluate your mteb compatible code by only changing a few lines!

## Links
| Binary |                                                                 Description                                                                |
|:------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| [FollowIR-7B](https://huggingface.co/jhu-clsp/FollowIR-7B) |   7B parameter model that does document reranking given a query and instructions. It is finetuned from Mistral-7B on the datasets below  | 
| [FollowIR-train](https://huggingface.co/datasets/jhu-clsp/FollowIR-train) | The dataset used to train FollowIR-7B. It consists of TREC instructions and queries, and GPT generated synthetic documents that have been filtered. |
| [FollowIR-train-raw](https://huggingface.co/datasets/jhu-clsp/FollowIR-train-raw) |  The pre-filtered version of the train set above. This was not used in model training as some GPT generated data is incorrect. |              

You can also find the individual annotated test data ([Robust04](https://huggingface.co/datasets/jhu-clsp/robust04-instructions), [Core17](https://huggingface.co/datasets/jhu-clsp/core17-instructions), and [News21](https://huggingface.co/datasets/jhu-clsp/news21-instructions)) although the format is best used with this evaluation code.

## Installation

```bash
git clone --recurse-submodules https://github.com/orionw/FollowIR.git
cd FollowIR/
conda create -n followir python=3.9 -y
conda activate followir
pip install -r requirements.txt
```

## Usage

You can use this in the same way you evaluate on MTEB -- the only change required is to check for `instructions` in kwargs of your `encode` function, and if so, to use them (they are given when encoding queries). Many examples for different types of models are in [models/](https://github.com/orionw/mteb-instruct/tree/master/models). If you have a model that already works with sentence_transformers, you can easily use it with no changes:

```bash
python -u models/base_sentence_transformer/evaluate_sentence_transformer.py --model_name_or_path 
```

If you have a different model (or want to use one that was evaluated in the paper) you can generate this python command with:

```bash
python evaluate_any.py --model MODEL_NAME
```

which will output the python command.

### Using a custom model

Models should implement the following interface, **exactly as in MTEB**, implementing an `encode` function taking as inputs a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.). **It must check for `instructions` in kwargs of the `encode` function and use them**. For inspiration, you can look at the [models/ section](https://github.com/orionw/mteb-instruct/tree/master/models) for a variety of model types.

```python
class MyModel():
    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if "instructions" in kwargs:
            # do something with the instructions
        pass

model = MyModel()
evaluation = MTEB(tasks=["Robust04InstructionReranking"])
evaluation.run(model)
```

Just like in the original MTEB, if you'd like to use different encoding functions for query and corpus when evaluating on Retrieval or Reranking tasks, you can add separate methods for `encode_queries` and `encode_corpus`. If these methods exist, they will be automatically used for those tasks.

```python
class MyModel():
    def encode_queries(self, queries, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            queries (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        if "instructions" in kwargs:
            # do something with the instructions...
        pass

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            corpus (`List[str]` or `List[Dict[str, str]]`): List of sentences to encode
                or list of dictionaries with keys "title" and "text"
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        pass
```

</details>

<br /> 


## Citing

If you found the code, data or model useful, free to cite:

```bibtex
@misc{weller2024followir,
      title={FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions}, 
      author={Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      year={2024},
      eprint={2403.15246},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

