import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einx

from jaxtyping import Float, Int, Float

from .nn import softmax


def cross_entropy(
    input: Float[Tensor, " [batch_size] vocab_size"], 
    target: Int[Tensor, " [batch_size]"]
) -> Float[Tensor, ""]:
    """
    Compute the cross-entropy loss for a batch of logits.

    Args:
        input (Float[Tensor, "[batch_size] vocab_size"]): The logits for each token in the vocabulary.
        target (Int[Tensor, "[batch_size]"]): The target indices for each example.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across all examples.
    """
    log_sum_exp = torch.logsumexp(input, dim=-1)
    target_logits = torch.gather(input, dim=-1, index=target.unsqueeze(-1))
    return torch.mean(log_sum_exp - target_logits)


def perplexity(
    input: Float[Tensor, " [batch_size] vocab_size"], 
    target: Int[Tensor, " [batch_size]"]
) -> Float[Tensor, ""]:
    return torch.exp(cross_entropy(input, target))