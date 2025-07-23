import einx
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
            )
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("... d_in, d_out d_in -> ... d_out", x, self.weight)
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"
    

def linear(weight: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    """ A linear function that can be used as a replacement for the Linear module. """
    return einx.dot("... d_in, d_out d_in -> ... d_out", in_features, weight)
    
    
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            )
        )
    
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.weight[x]
    

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, device=device, dtype=dtype),
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.

        Args:
            x: The input tensor of shape (..., d_model).

        Returns:
            The output tensor of shape (..., d_model).
        """
        denom = einx.sum("... [d_model]", x**2) / self.d_model + self.eps
        return x / torch.sqrt(denom)[..., None] * self.weight
    

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
@torch.compile
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
    

class GLU(nn.Module):
    """ Gated Linear Unit """
    def __init__(self, dim=-1):
        """
        Args:
            dim: The dimension to split the input tensor into two parts.
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=self.dim)
        return a * torch.sigmoid(b)


class FFN(nn.Module):
    """ FFN with SwiGLU. 
    
    $ FFN(x) = SwiGLU(x, W1, W2, W3) = W2 @ (SiLU(W1 @ x) \\circ (W3 @ x)) $ 
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self.find_d_ff(d_model)
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        
    @staticmethod
    def find_d_ff(d_model: int, d_ff_ratio: float = 8/3) -> int:
        """ Approximate the d_ff parameter for a FFN layer by (d_model * d_ff_ratio) but rounded to the nearest multiple of 64. """
        return int(d_ff_ratio * d_model) // 64 * 64
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model: The dimension of the model.
            max_seq_len: The maximum sequence length.
            theta: The base frequency.
            device: The device to store the weights.
            dtype: The dtype of the weights.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        assert d_model % 2 == 0, "d_model must be even"
        
        half_dim = d_model // 2
        freq_seq = torch.arange(half_dim, device=device, dtype=dtype)
        inv_freq = 1.0 / (theta ** (freq_seq / half_dim)) # shaped (d_model // 2, )
        
        pos = torch.arange(max_seq_len, device=device, dtype=dtype)
        freqs = torch.outer(pos, inv_freq) # shaped (max_seq_len, d_model // 2)   
        # freqs = einx.dot("i, j -> ij", pos, inv_freq)
        
        self.register_buffer("cos_cached", torch.cos(freqs)[None, :, :], persistent=False) # shaped (1, max_seq_len, d_model // 2)
        self.register_buffer("sin_cached", torch.sin(freqs)[None, :, :], persistent=False) # shaped (1, max_seq_len, d_model // 2)
        
    def forward(self, x: torch.Tensor, token_positions: torch.LongTensor | None = None) -> torch.Tensor:
        """
        Args:
            x: The input tensor of shape (..., seq_len, d_model).
            token_positions: The token positions of shape (..., seq_len).

        Returns:
            The output tensor of shape (..., seq_len, d_model).
        """
        if token_positions is None:
            seq_len = x.shape[-2]
            if seq_len > self.max_seq_len:
                raise ValueError(f"seq_len ({seq_len}) is greater than max_seq_len ({self.max_seq_len})")
            cos_cached, sin_cached = (
                self.cos_cached[..., :seq_len, :],
                self.sin_cached[..., :seq_len, :] # shaped (1, seq_len, d_model // 2)
            )
        else:
            seq_len = token_positions.shape[-1]
            if torch.any(token_positions > self.max_seq_len):
                raise ValueError(f"token_positions ({token_positions}) is greater than max_seq_len ({self.max_seq_len})")
            cos_cached, sin_cached = (
                self.cos_cached[..., token_positions, :],
                self.sin_cached[..., token_positions, :] # shaped (1, seq_len, d_model // 2)
            )
        
        x0, x1 = x[..., ::2], x[..., 1::2]
       
        x0, x1 = x0 * cos_cached - x1 * sin_cached, x0 * sin_cached + x1 * cos_cached
        return torch.stack([x0, x1], dim=-1).flatten(start_dim=-2)
        
        
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_rectified = x - x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x_rectified)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        q: The query tensor of shape (..., queries, d_k).
        k: The key tensor of shape (..., keys, d_k).
        v: The value tensor of shape (..., values, d_v).
        mask: The mask tensor of shape (..., queries, keys).
    Returns:
        The output tensor of shape (..., queries, d_v).
    """
    d_k = Q.shape[-1]
    attn_scores = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K) / d_k ** 0.5
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    attn_probs = softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, V)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        causal: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope_theta: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.causal = causal
        
        if rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(
                d_model, max_seq_len, theta=rope_theta, device=device, dtype=dtype,
            )
        
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        
        num_seq_len = x.shape[-2]
        if self.causal:
            # Create causal mask: True for positions that should be attended to
            mask = torch.tril(
                torch.ones(num_seq_len, num_seq_len, device=x.device, dtype=x.dtype),
                diagonal=0,
            ).bool()
        else:
            mask = None
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = einx.rearrange("b l (h d) -> b h l d", q, h=self.num_heads)
        k = einx.rearrange("b l (h d) -> b h l d", k, h=self.num_heads)
        v = einx.rearrange("b l (h d) -> b h l d", v, h=self.num_heads)
        
        if hasattr(self, "rope"):
            # RoPE should be applied to each head dimension, not the full model dimension
            rope = RotaryPositionalEmbedding(
                self.d_head, self.max_seq_len, theta=self.rope.theta, 
                device=x.device, dtype=x.dtype
            )
            q = rope(q, token_positions)
            k = rope(k, token_positions)
                
        return self.output_proj(
            einx.rearrange(
                "b h l d -> b l (h d)",
                scaled_dot_product_attention(q, k, v, mask),
            )
        )
        

class TransformerBlock(nn.Module):
    """ 
    A transformer block with a multi-headed self-attention and a feedforward network.
    
    First sublayer:
    y = x + MultiHeadSelfAttention(RMSNorm(x))
    
    Second sublayer:
    y = y + FFN(RMSNorm(y))
    
    
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        causal: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope_theta: float | None = None,
    ):
        """
        Args:
            d_model: The dimension of the model.
            num_heads: The number of attention heads.
            d_ff: The dimension of the feedforward network.
            max_seq_len: The maximum sequence length.
            causal: Whether to use a causal mask.
            device: The device to store the weights.
            dtype: The dtype of the weights.
            rope_theta: The base frequency for RoPE.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.device = device
        
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(
            d_model, num_heads, max_seq_len, causal, device, dtype, rope_theta,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FFN(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x