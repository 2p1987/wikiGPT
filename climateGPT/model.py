# Taken from Llama (facebook research repo) and llama.c from Andrej Karpathy's repo.
# The version is slightly simplified: we get rid of the kv cache and mutli-group query
# attention since we aim at training small language models.

import inspect
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import structlog
import torch
from torch import nn
from torch.nn import functional as F

log = structlog.get_logger()

# model params


@dataclass
class MoeArgs:
    """
    A class used to store arguments for a sparse mixture of experts model.

    ...

    Attributes
    ----------
    num_experts : int
        number of experts in the model
    num_experts_per_tok : int
        number of experts per token in the model
    """

    enable_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 1
    moe_dropout: float = 0.0


@dataclass
class ModelArgs:
    """
    A class used to store arguments for a model.

    ...

    Attributes
    ----------
    dim : int
        dimension of the model embeddings
    n_layers : int
        number of layers in the model (transformer blocks + FF)
    n_heads : int
        number of attention heads for queries
    n_kv_heads: int
        number of attention heads for keys and values (must be a divisor of n_heads)
    vocab_size : int
        size of the vocabulary
    hidden_dim : int
        dimension of the hidden layer
    hidden_dim_multiplier : int
        multiplier for the hidden layer dimension
    multiple_of : int
        MLP hidden layer size will be multiple of this value
    norm_eps : float
        a small number added for stability in normalization
    max_context_length : int
        maximum context length for the model
    dropout : float
        dropout rate for the model
    moe : Optional[MoeArgs]
        arguments for the sparse mixture of experts model
    """

    moe: MoeArgs
    dim: int = 4096
    n_layers: int = 1
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
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
    """
    Reshapes the frequency tensor for broadcasting.

    This function reshapes the frequency tensor to match the shape of the input tensor x
    but with dimensions set to 1 except for the second and last dimensions. This is done
    to enable broadcasting when performing operations between freqs_cis and x.

    Parameters
    ----------
    freqs_cis : torch.Tensor
        The frequency tensor to be reshaped.
    x : torch.Tensor
        The input tensor whose shape is used as a reference for reshaping
        (expected batch_size, context_length, n_head, head_dim).

    Returns
    -------
    torch.Tensor
        The reshaped frequency tensor.

    Raises
    ------
    AssertionError
        If the number of dimensions of x is less than or equal to 1, or if the shape of
        freqs_cis does not match the second and last dimensions of x.
    """
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional encoding to the input tensors embeddings.

    This function applies rotary positional encoding to the input tensors xq and xk
    using the provided cosine and sine frequencies. The input tensors are first reshaped
    to match the complex representation (first position is real part, second is
    imaginary part for each sequential pair in the embedding vector), then the
    frequencies are reshaped for broadcasting. The rotation is applied and the last two
    dimensions are flattened back in their original order.

    Parameters
    ----------
    xq : torch.Tensor
        The input tensor for query.
    xk : torch.Tensor
        The input tensor for key.
    freqs_cos : torch.Tensor
        The cosine frequencies for rotation.
    freqs_sin : torch.Tensor
        The sine frequencies for rotation.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The output tensors after applying rotary positional encoding.

    """
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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the tensor values n times to match dimension of the projected queries.
    """
    batch_size, context_length, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, context_length, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, context_length, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    A class used to implement the Attention mechanism in a transformer model.

    ...

    Attributes
    ----------
    n_heads : int
        number of attention heads for queries
    n_kv_heads : int
        number of attention heads for keys and values (must be a divisor of n_heads)
    dim : int
        embedding dimension of the model
    head_dim : int
        dimension of each attention head
    wq : torch.nn.Linear
        linear layer for query
    wk : torch.nn.Linear
        linear layer for key
    wv : torch.nn.Linear
        linear layer for value
    wo : torch.nn.Linear
        final linear layer for output
    dropout : float
        dropout rate for the model
    attn_dropout : torch.nn.Dropout
        dropout layer for attention
    resid_dropout : torch.nn.Dropout
        dropout layer for residual connection
    flash : bool
        flag to use flash attention if available

    Methods
    -------
    forward(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor)
        -> torch.Tensor:
        Passes the input through the Attention layer.
    """

    def __init__(self, args: ModelArgs) -> None:
        """
        Constructs all the necessary attributes for the Attention object.

        Parameters
        ----------
            args : ModelArgs
                model arguments containing parameters
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert (
            args.n_heads % self.n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads"
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dim = args.dim
        self.head_dim = self.dim // args.n_heads
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
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
        """
        Passes the input through the Attention layer.

        Parameters
        ----------
            x : torch.Tensor
                input tensor
            freqs_cos : torch.Tensor
                pre-computed cosine frequencies for rotary positional encoding
            freqs_sin : torch.Tensor
                prer-computed sine frequencies for rotary positional encoding

        Returns
        -------
            torch.Tensor
                output tensor after passing through the Attention layer
        """
        batch_size, context_length, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # x{q, k, v}: (batch_size, context_length, dim) @ (dim, dim)
        # --> (batch_size, context_length, dim)
        xq = xq.view(batch_size, context_length, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, context_length, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, context_length, self.n_kv_heads, self.head_dim)
        # xq: (batch_size, context_length, n_heads, head_dim)
        # x{k, v}: (batch_size, context_length, n_kv_heads, head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        # (batch_size, context_length, n_heads, head_dim)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

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


# FF module
class FeedForward(nn.Module):
    """
    A class used to implement the FeedForward mechanism in a transformer model.

    ...

    Attributes
    ----------
    w1 : torch.nn.Linear
        first linear layer
    w2 : torch.nn.Linear
        second linear layer
    w3 : torch.nn.Linear
        third linear layer
    dropout : torch.nn.Dropout
        dropout layer

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Passes the input through the FeedForward layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_dim_multiplier: int,
        multiple_of: int,
        dropout: float,
    ):
        """
        Constructs all the necessary attributes for the FeedForward object.

        Parameters
        ----------
            dim : int
                dimension of the model
            hidden_dim : int
                dimension of the hidden layer
            hidden_dim_multiplier : int
                multiplier for the hidden layer dimension
            multiple_of : int
                MLP hidden layer size will be multiple of this value
            dropout : float
                dropout rate for the model
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim_multiplier * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the FeedForward layer.

        Parameters
        ----------
            x : torch.Tensor
                input tensor

        Returns
        -------
            torch.Tensor
                output tensor after passing through the FeedForward layer
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoeLayer(nn.Module):
    """
    A class used to implement a sparse mixture of experts layer in a transformer model.

    ...

    Attributes
    ----------
    experts : nn.ModuleList
        list of experts in the model
    gate : nn.Module
        gating mechanism for the model (simple linear layer)
    args : MoeArgs

    Methods
    -------
    forward(inputs: torch.Tensor) -> torch.Tensor:
        Passes the input through the sparse mixture of experts layer.
    """

    def __init__(self, experts: List[FeedForward], gate: nn.Module, moe_args: MoeArgs):
        """
        Construct all the necessary attributes for the MoeLayer object.

        Parameters
        ----------
            experts : List[nn.Module]
                list of experts in the model
            gate : nn.Module
                gating mechanism for the model (simple linear layer)
            moe_args : MoeArgs
                arguments for the sparse mixture of experts model
        """
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args
        self.token_load_balancing = {i: 0 for i in range(self.args.num_experts)}

    def forward(self, x: torch.Tensor):
        """
        Passes the input through the sparse mixture of experts layer.

        Parameters
        ----------
            x : torch.Tensor
                input tensor

        Returns
        -------
            torch.Tensor
                output tensor after passing through the sparse mixture of experts layer
        """
        inputs_squashed = x.view(-1, x.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(x)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):  # this loop is the bottleneck
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
            self.token_load_balancing[i] += len(batch_idx)

        return results.view_as(x)


# Transformer block
class TransformerBlock(nn.Module):
    """
    A class used to implement a Transformer block in a transformer model.

    ...

    Attributes
    ----------
    layer_id : int
        identifier for the layer
    attn : Attention
        attention mechanism for the transformer block
    feedforward : FeedForward
        feedforward mechanism for the transformer block
    attn_norm : RMSNorm
        normalization for the attention mechanism
    ff_norm : RMSNorm
        normalization for the feedforward mechanism

    Methods
    -------
    forward(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor)
    -> torch.Tensor:
        Passes the input through the Transformer block.
    """

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        enable_moe: bool = False,
    ) -> None:
        """
        Constructs all the necessary attributes for the TransformerBlock object.

        Parameters
        ----------
            layer_id : int
                identifier for the layer
            args : ModelArgs
                model arguments containing various parameters
            activate_moe : bool
                flag to activate the sparse mixture of experts model
        """
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(args)
        self.enable_moe = enable_moe

        if self.enable_moe:
            self.feed_forward = MoeLayer(
                experts=[
                    FeedForward(
                        dim=args.dim,
                        hidden_dim=args.hidden_dim,
                        hidden_dim_multiplier=args.hidden_dim_multiplier,
                        multiple_of=args.multiple_of,
                        dropout=args.moe.moe_dropout,
                    )
                    for _ in range(args.moe.num_experts)
                ],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(
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
    ):
        """
        Passes the input through the Transformer block.

        Parameters
        ----------
            x : torch.Tensor
                input tensor
            freqs_cos : torch.Tensor
                cosine frequencies for rotary positional encoding
            freqs_sin : torch.Tensor
                sine frequencies for rotary positional encoding

        Returns
        -------
            torch.Tensor
                output tensor after passing through the Transformer block
        """
        x_attn_norm = self.attn_norm(x)
        h = x + self.attn(x_attn_norm, freqs_cos, freqs_sin)  # residual connection here
        x_ff_norm = self.ff_norm(h)
        output = h + self.feed_forward(x_ff_norm)  # residual connection here

        return output


# Transformer model
class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor] = None

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.enable_moe = args.moe.enable_moe
        if self.enable_moe:
            self.load_balancing = {i: 0 for i in range(args.moe.num_experts)}

        # Define model layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id,
                    args,
                    self.enable_moe,
                )
            )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # share the embedding parameters with the output parameters
        self.tok_embeddings.weight = self.output.weight
        # https://paperswithcode.com/method/weight-tying

        # pre-compute freqs cos and sin for the rotary positional encoding
        freqs_cos, freqs_sin = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_context_length
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, context_length = tokens.shape
        freqs_cos = self.freqs_cos[:context_length]
        freqs_sin = self.freqs_sin[:context_length]

        # embed tokens and positions
        tokens = self.tok_embeddings(tokens)
        # token: (batch_size, context_length, dim)

        # apply dropout to the token embeddings
        tokens = self.dropout(tokens)

        # transformer layers
        for layer in self.layers:
            tokens = layer(tokens, freqs_cos, freqs_sin)

        # normalize and project to output vocab
        tokens = self.norm(tokens)

        if targets is not None:
            logits = self.output(tokens)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the output on the very last
            #  position
            logits = self.output(
                tokens[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def configure_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
        device_type: str,
    ):
        # start with all candidate parameters
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed
        # , otherwise no.
        # i.e. all weight tensors in matmuls + embeddings are decayed, all biases and
        # layernorms aren't.
        decay_params = [p for _, p in params.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in params.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log.info(
            f"""
            Number of decayed parameter tensors: {len(decay_params)}
            Containing {num_decay_params:,} parameters
            """
        )
        log.info(
            f"""
            Number of non-decayed parameter tensors: {len(nodecay_params)}
            Containing {num_nodecay_params:,} parameters
            """
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            params=optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        log.info(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt, flops_promised=312e12):
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.args
        L, H, Q, T = (
            cfg.n_layers,
            cfg.n_heads,
            cfg.dim // cfg.n_heads,
            cfg.max_context_length,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = flops_promised
        # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # M1 MPS GPU float16 peak flops is 2.6 TFLOPS

        mfu = flops_achieved / flops_promised
        return mfu

    def get_token_load_balancing_state(self):
        if self.enable_moe:
            total_token_balance = {i: 0 for i in range(self.args.moe.num_experts)}
            for i in range(self.args.moe.num_experts):
                for layer in self.layers:
                    total_token_balance[i] += layer.feed_forward.token_load_balancing[i]
            total = sum(total_token_balance.values())
            total_token_balance = {
                i: round(v / total * 100, 1) for i, v in total_token_balance.items()
            }
            return total_token_balance
        else:
            return None

    @torch.inference_mode()  # TODO: implement with k/v cache?
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and
        complete the sequence max_new_tokens times, feeding the predictions back into
        the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for
          this.
        Also note this is a super inefficient version of sampling with no key/value
        cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.args.max_context_length
                else idx[:, -self.args.max_context_length :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    model_args = ModelArgs()
    tokens = torch.randint(low=0, high=300, size=(1, 10))  # create random int
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    model = Transformer(model_args)
    model.configure_optimizer(0.002, 0.1, (0.9, 0.95), "cpu")
    model.eval()
    for idx in model.generate(x, 10):
        print(idx)
