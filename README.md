<h1 align="center">FollowIR: using MTEB -- <b>with Instructions</b></h1>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="#leaderboard">Leaderboard</a> |
        <a href="#citing">Citing</a> |
    <p>
</h4>


## Installation

```bash
pip install git+https://github.com/orionw/mteb-instruct.git
```

## Usage

* Using a python script (see [models/](https://github.com/orionw/mteb-instruct/tree/master/models) for several examples). You can use this in the same way you evaluate on MTEB -- the only change required is to check for "instructions" in kwargs of your `encode` function, and if so, to use them. If you have a model that already works with sentence_transformers, you can use it like this:

```bash
python -u models/base_sentence_transformer/evaluate_sentence_transformer.py --model_name_or_path 
```

You can generate this python command for any of the models we have already added with:

```bash
python evaluate_any.py --model MODEL_NAME
```

which will output the python command.

### Using a custom model

Models should implement the following interface, **exactly as in MTEB**, implementing an `encode` function taking as inputs a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.). **It must check for `instructions` in kwargs of the `encode` function and use them. For inspiration, you can look at the [models/ section](https://github.com/orionw/mteb-instruct/tree/master/models) for a variety of model types.

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

FollowIR was introduced in TODO, feel free to cite:

```bibtex
TODO
```

