import torch
import logging

from torch import Tensor
from transformers import PreTrainedTokenizerFast, BatchEncoding
from typing import Mapping, Dict, List


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
        attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
        s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        emb = s / d
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def create_batch_dict(tokenizer: PreTrainedTokenizerFast, input_texts: List[str], always_add_eos: bool, max_length: int = 512) -> BatchEncoding:
    if not always_add_eos:
        return tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )
    else:
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True
        )

        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]

        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

