"""
Assorted utilities for working with neural networks by Sanxing Chen (sc3hn@virginia.edu)
"""

import torch
from torch import nn
from torch.nn import functional as F


def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    Based on https://github.com/allenai/allennlp #5ad7a33
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor

def get_randperm_from_lengths(sequence_lengths: torch.Tensor, max_length: int):
    rand_vector = sequence_lengths.new_empty((len(sequence_lengths), max_length), dtype=torch.float).uniform_(0.1, 1)
    rand_vector.masked_fill_(~get_mask_from_sequence_lengths(sequence_lengths, max_length), 0)
    perm_vector = rand_vector.argsort(descending=True)
    return perm_vector

def get_bernoulli_mask(shape, prob, device):
    return torch.bernoulli(torch.full(shape, prob, device=device)).bool()