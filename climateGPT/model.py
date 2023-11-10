# Taken from Llama (facebook research repo) and llama.c from Andrej Karpathy's repo.
# The version is slightly simplified: we get rid of the kv cache and mutli-group query
# attention since we aim at training small language models.

import math
from dataclasses import dataclass
from typing import Tuple

import structlog
import torch
from torch import nn
from torch.nn import functional as F

log = structlog.get_logger()

# model params


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 1
    n_heads: int = 32
    vocab_size: int = 32000
    hidden_dim: int = 4096
    hidden_dim_multiplier: int = 2
    multiple_of: int = 16  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_context_length: int = 2048
    dropout: float = 0.0


# Rotary Positional Encoding (RoPE) utils


def precompute_freqs_cis(
    dim: int, context_length: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-computes cosines and sines of different frequencies.
    These are used to rotate the embeddings in the apply_rotary_emb method.

    Parameters
    ----------
        dim : int
            dimension of the embeddings (expected to be even)
        context_length : int
            number of positions to be encoded. The frequency is multiplied by this
            number
        theta : float
            a parameter controlling the range of frequencies. The higher the theta,
            the wider the range of frequencies

    Returns
    -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the cosines and sines of different frequencies

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(context_length, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# RMS module
class RMSNorm(nn.Module):
    """
    A class used to implement Root Mean Square Layer Normalization.

    ...

    Attributes
    ----------
    eps : float
        a small number added for stability
    weight : torch.nn.Parameter
        learnable scale parameters

    Methods
    -------
    _norm(x: torch.Tensor) -> torch.Tensor:
        Normalizes the input tensor using RMSNorm.
    forward(x):
        Passes the input through the RMSNorm layer.
    """

    def __init__(self, dim: int, eps: float) -> None:
        """
        Constructs all the necessary attributes for the RMSNorm object.

        Parameters
        ----------
            dim : int
                dimension of the input tensor
            eps : float
                a small number added for stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor using RMSNorm.

        Parameters
        ----------
            x : torch.Tensor
                input tensor

        Returns
        -------
            torch.Tensor
                normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Passes the input through the RMSNorm layer.

        Parameters
        ----------
            x : input tensor

        Returns
        -------
            output tensor after passing through the RMSNorm layer
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# Attention module
class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // args.n_heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.dropout = args.dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # use flash attention if available
        self.flash = hasattr(nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            log.warn("Using slow attention. Flash attention requires PyTorch >= 2.0")
            mask = torch.full(
                (1, 1, args.max_context_length, args.max_context_length), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ) -> torch.Tensor:
        batch_size, context_length, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # x{q, k, v}: (batch_size, context_length, dim) @ (dim, dim)
        # --> (batch_size, context_length, dim)
        xq = xq.view(batch_size, context_length, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, context_length, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, context_length, self.n_heads, self.head_dim)
        # x{q, k, v}: (batch_size, context_length, n_heads, head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # x{q, k, v}: (batch_size, n_heads, context_length, head_dim)
        # --> (batch_size, n_heads, context_length, head_dim)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = (
                scores + self.mask[:, :, :context_length, :context_length]
            )  # (batch_size, n_heads, context_length, cache_len + context_length)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(
                scores, xv
            )  # (batch_size, n_heads, context_length, head_dim)

        # restore context_length as batch dimension and concat heads
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view((batch_size, context_length, self.dim))
        )  # contiguous ensure that the tensor is stored in a contiguous chunk of memory
        # this is necessary so that the view() method can operate without any issue

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output


if __name__ == "__main__":
    model_args = ModelArgs()
    freqs_cos, freqs_sin = precompute_freqs_cis(
        model_args.dim // model_args.n_heads, model_args.max_context_length
    )
    attention = Attention(ModelArgs())
    attention(
        torch.rand(1, model_args.max_context_length, model_args.dim),
        freqs_cos,
        freqs_sin,
    )


# FF module
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_dim_multiplier: int,
        multiple_of: int,
        dropout: float,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim_multiplier * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(args)
        self.feedforward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            hidden_dim_multiplier=args.hidden_dim_multiplier,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ff_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ) -> torch.Tensor:
        x_attn_norm = self.attn_norm(x)
        h = x + self.attn(x_attn_norm, freqs_cos, freqs_sin)  # residual connection here
        x_ff_norm = self.ff_norm(h)
        output = h + self.feedforward(x_ff_norm)  # residual connection here

        return output


# Transformer model
