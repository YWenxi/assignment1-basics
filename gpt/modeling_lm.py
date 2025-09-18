import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int


from .nn import TransformerBlock, Embedding, Linear, RMSNorm
from .nn import softmax


class CausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, True, device, dtype, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, 
                in_indices: Int[Tensor, " batch_size sequence_length"],
                ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        """
        Args:
            in_indices (Int[Tensor, " batch_size sequence_length"]): The input indices to the language model.

        Returns:
            Float[Tensor, " batch_size sequence_length vocab_size"]: The logits of the language model.
        """
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    
    
def GPT2_XL():
    return CausalLM(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        rope_theta=10000.0,
    )


def gpt2():
    return CausalLM(
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072, # 4 * d_model
        rope_theta=10000.0,
    )


all_models = {
    "gpt2-xl": GPT2_XL,
    "gpt2": gpt2,
}