<h1 align="center">FollowIR: Evaluating and Teaching Information
Retrieval Models to Follow Instructions</b></h1>

<h4 align="center">
    <p>
        <a href="#links">Model/Data Links</a> |
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="https://huggingface.co/spaces/mteb/leaderboard?task=InstructionRetrieval">Leaderboard</a> |
        <a href="#citing">Citing</a> |
    <p>
</h4>

Official repository for the paper [FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions](https://arxiv.org/abs/2403.15246). Official evaluation can be done by installing the `mteb` library and evaluating your MTEB compatible model with zero (or only a few) lines of code changes!

## Links
| Binary |                                                                 Description                                                                |
|:------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| [FollowIR-7B](https://huggingface.co/jhu-clsp/FollowIR-7B) |   7B parameter model that does document reranking given a query and instructions. It is finetuned from Mistral-7B on the datasets below  | 
| [FollowIR-train](https://huggingface.co/datasets/jhu-clsp/FollowIR-train) | The dataset used to train FollowIR-7B. It consists of TREC instructions and queries, and GPT generated synthetic documents that have been filtered. |
| [FollowIR-train-raw](https://huggingface.co/datasets/jhu-clsp/FollowIR-train-raw) |  The pre-filtered version of the train set above. This was not used in model training as some GPT generated data is incorrect. |              

You can also find the individual annotated test data ([Robust04](https://huggingface.co/datasets/jhu-clsp/robust04-instructions), [Core17](https://huggingface.co/datasets/jhu-clsp/core17-instructions), and [News21](https://huggingface.co/datasets/jhu-clsp/news21-instructions)) although the format is best used with MTEB's evaluation code.

## Installation 
If you wish to reproduce the experiments in the paper you can use the following code:

```bash
git clone https://github.com/orionw/FollowIR.git
cd FollowIR/
conda create -n followir python=3.9 -y
conda activate followir
pip install -r requirements.txt
bash launch_all_jobs.sh
```

## Usage 
If your model is `SentenceTransformer` compatible and requires no special tokens for concatenating the query and instructions, you can simply use the following one line command: 
```bash
mteb -m $MODEL_NAME -t $DATASET
```
for each of the datasets in `{Robust04InstructionRetrieval, Core17InstructionRetrieval, News21InstructionRetrieval}`

If you have a bi-encoder model but want to do something different than simply appending the instruction to the query with a space, you can extend `DenseRetrievalExactSearch` and check for `instructions` in kwargs. See (see [models/base_sentence_transformers/](https://github.com/orionw/mteb-instruct/tree/master/models/base_sentence_transformers) as a starting place for small modifiations and [models/e5/](https://github.com/orionw/mteb-instruct/tree/master/models/e5/evaluate_e5.py) for an example with larger modifications).

### Reranker Usage

Rerankers have now been added to MTEB! If you are using a reranker model, you will need to extend the `DenseRetrievalExactSearch` class and define an `__init__` and `predict` function (see [models/rerankers section](https://github.com/orionw/mteb-instruct/tree/master/models/rerankers/reranker_models.py) for a variety of reranker examples). Your predict function should take in `input_to_rerank` which will be a tuple of the form:
```python
# if there are no instructions, instructions will be a list of Nones
# Instructions will be present for all of the FollowIR datasets
queries, passages, instructions = list(zip(*input_to_rerank))
```

Your `predict` function should use these and return a list containing a score for each tuple item.


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

