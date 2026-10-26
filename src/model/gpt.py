import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self, hidden_dim: int, expansion_factor: int, dropout: float, bias: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor

        self.expand = nn.Linear(hidden_dim, expansion_factor * hidden_dim, bias=bias)
        self.project = nn.Linear(expansion_factor * hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Tensor: (batch, seq_len, hidden_dim)

        Returns:
            Tensor: (batch, seq_len, hidden_dim)
        """

        assert x.dim() == 3, "Input should be 3d tensor"
        assert x.shape[-1] == self.hidden_dim, "Shapes mismatch"

        x = self.expand(x)
        x = F.gelu(x)
        x = self.project(x)
        x = self.dropout(x)

        return x


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_bias: bool,
        attn_dropout: float,
        resid_dropout: float,
        use_xsa: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()

        assert num_heads > 0
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=attn_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

        self.attn_dropout = attn_dropout
        self.resid_dropout = nn.Dropout(resid_dropout)

        self.use_xsa = use_xsa
        self.rope = RoPE(self.head_dim) if use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Tensor: (batch, seq_len, hidden_dim)

        Returns:
            Tensor: (batch, seq_len, hidden_dim)
        """
        assert x.dim() == 3, "Input should be 3d tensor"
        assert x.shape[-1] == self.hidden_dim, "Shapes mismatch"

        B, S, H = x.shape

        q, k, v = self.qkv_proj(x).split(self.hidden_dim, dim=2)

        q = q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0,
            is_causal=True,
        )

        if self.use_xsa:
            v_n = torch.nn.functional.normalize(v, dim=-1)
            out = out - (out * v_n).sum(dim=-1, keepdim=True) * v_n

        out = out.transpose(1, 2).contiguous().view(B, S, H)

        out = self.resid_dropout(self.out_proj(out))

        return out


class GPTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_bias: bool,
        ln_bias: bool,
        attn_dropout: float,
        resid_dropout: float,
        mlp_expansion_factor: int,
        mlp_dropout: float,
        use_xsa: bool = False,
        use_rope: bool = False,
        mlp_bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.ln1 = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.attn = CausalSelfAttention(
            hidden_dim,
            num_heads,
            attn_bias,
            attn_dropout,
            resid_dropout,
            use_xsa=use_xsa,
            use_rope=use_rope,
        )
        self.ln2 = nn.LayerNorm(hidden_dim, bias=ln_bias)
        self.mlp = MLP(hidden_dim, mlp_expansion_factor, mlp_dropout, bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Tensor: (batch, seq_len, hidden_dim)

        Returns:
            Tensor: (batch, seq_len, hidden_dim)
        """
        assert x.dim() == 3, "Input should be 3d tensor"
        assert x.shape[-1] == self.hidden_dim, "Shapes mismatch"

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class RoPE(nn.Module):
    """
    https://wangkuiyi.github.io/rope.html
    """

    def __init__(self, hidden_dim: int, base: float = 10000):
        super().__init__()

        assert hidden_dim % 2 == 0, "RoPE requires an even head dimension"
        self.hidden_dim = hidden_dim

        theta = 1.0 / (base ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Tensor: (..., seq_len, hidden_dim)

        Returns:
            Tensor: (..., seq_len, hidden_dim)
        """
        assert x.dim() in (3, 4), "Input should be 3d or 4d tensor"
        assert x.shape[-1] == self.hidden_dim, "Shapes mismatch"

        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.theta)
        theta = torch.einsum("i , j -> i j", t, self.theta)

        emb = torch.cat((theta, theta), dim=-1)
        sin = torch.sin(emb).to(x.dtype)
        cos = torch.cos(emb).to(x.dtype)

        xu, xd = x[..., : self.hidden_dim // 2], x[..., self.hidden_dim // 2 :]
        hatx = torch.cat((-xd, xu), dim=-1)

        return x * cos + hatx * sin


class GPT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        block_size: int,
        hidden_dim: int,
        num_heads: int,
        attn_bias: bool,
        ln_bias: bool,
        emb_dropout: float,
        attn_dropout: float,
        resid_dropout: float,
        mlp_expansion_factor: int,
        mlp_dropout: float,
        token_emb_ln: bool = False,
        use_rope: bool = False,
        use_xsa: bool = False,
        mlp_bias: bool = True,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.vocab_size, self.hidden_dim),
                wpe=(
                    nn.Embedding(self.block_size, self.hidden_dim)
                    if not use_rope
                    else None
                ),
                drop=nn.Dropout(emb_dropout),
                h=nn.ModuleList(
                    [
                        GPTBlock(
                            hidden_dim,
                            num_heads,
                            attn_bias,
                            ln_bias,
                            attn_dropout,
                            resid_dropout,
                            mlp_expansion_factor,
                            mlp_dropout,
                            use_xsa=use_xsa,
                            use_rope=use_rope,
                            mlp_bias=mlp_bias,
                        )
                        for _ in range(n_layers)
                    ]
                ),
                final_ln=nn.LayerNorm(self.hidden_dim, bias=ln_bias),
            )
        )

        if token_emb_ln:
            self.transformer.wte = nn.Sequential(
                self.transformer.wte, nn.LayerNorm(self.hidden_dim, bias=ln_bias)
            )

        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # https://paperswithcode.com/method/weight-tying
        embedding = self.transformer.wte[0] if token_emb_ln else self.transformer.wte
        self.lm_head.weight = embedding.weight

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith(("attn.out_proj.weight", "mlp.project.weight")):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, input_ids: torch.Tensor, **batch) -> torch.Tensor:
        """
        Args:
            Tensor: (batch, seq_len)

        Returns:
            logits (dict): dict containing calculated logits (batch, seq_len, vocab_size)
        """
        assert input_ids.dim() == 2, "Input should be 2d tensor"
        assert input_ids.dtype == torch.long, "Input should be long tensor"

        B, S = input_ids.shape

        assert 0 < S <= self.block_size, "seq_len exceeds block_size"

        token_embs = self.transformer.wte(input_ids)

        if self.transformer.wpe is not None:
            pos = torch.arange(0, S, dtype=torch.long, device=input_ids.device)
            pos_embs = self.transformer.wpe(pos)
            token_embs = token_embs + pos_embs

        x = self.transformer.drop(token_embs)

        for layer in self.transformer.h:
            x = layer(x)

        x = self.transformer.final_ln(x)
        x = self.lm_head(x)

        return {"logits": x}

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
