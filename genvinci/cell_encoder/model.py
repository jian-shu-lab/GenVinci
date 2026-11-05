from __future__ import annotations

import os
import json
from typing import Optional

import torch
import torch.nn as nn


PAD_TOKEN_ID = 1
CLS_TOKEN_ID = 0
CONTEXT_GENE_TOKEN_IDS = (3, 4, 5, 6)  # modality, assay, organism, tissue
N_PREFIX_TOKENS = 1 + len(CONTEXT_GENE_TOKEN_IDS)  # CLS + context


class GenVinci(nn.Module):
    """GenVinci single-cell foundation model (encoder only)."""

    def __init__(self,
                 dim_model: int = 512,
                 nheads: int = 8,
                 dim_feedforward: int = 2048,
                 nlayers: int = 12,
                 n_gene_tokens: int = 25030,
                 n_expression_tokens: int = 192,
                 max_seq_len: int = 1024,
                 num_bins: int = 50,
                 dropout: float = 0.1,
                 cls_classes: Optional[int] = None,
                 **kwargs
                ):

        super().__init__()

        self.config = dict(
            dim_model=dim_model,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            nlayers=nlayers,
            n_gene_tokens=n_gene_tokens,
            n_expression_tokens=n_expression_tokens,
            max_seq_len=max_seq_len,
            num_bins=num_bins,
            dropout=dropout,
            cls_classes=cls_classes,
        )

        # Core transformer components
        self._init_transformer(dim_model, nheads, dim_feedforward, nlayers, dropout)

        # Embedding layers
        self._init_embeddings(n_gene_tokens, dim_model, n_expression_tokens, dropout)

        # Task-specific head for fine-tuning
        if cls_classes:
            self._init_cls_heads(dim_model, cls_classes)


    def _init_transformer(self, dim_model: int, nheads: int, dim_feedforward: int,
                         nlayers: int, dropout: float) -> None:
        """Initialize transformer components."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation='gelu',
            layer_norm_eps=1e-12
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=nlayers,
            enable_nested_tensor=False
        )

    def _init_embeddings(self, n_gene_tokens: int, dim_model: int, n_expression_tokens: int,
                        dropout: float) -> None:
        """Initialize embedding layers."""

        self.gene_embeddings = nn.Embedding(
            num_embeddings=n_gene_tokens,
            embedding_dim=dim_model,
            padding_idx=PAD_TOKEN_ID
        )

        self.expression_embeddings = nn.Embedding(
            num_embeddings=n_expression_tokens,
            embedding_dim=dim_model,
            padding_idx=PAD_TOKEN_ID
        )

        # 0: CLS token, 1: context tokens, 2: gene tokens
        self.segment_embeddings = nn.Embedding(
            num_embeddings=3,
            embedding_dim=dim_model,
        )

        self.layer_norm = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(p=dropout)

    def _init_cls_heads(self, dim_model: int, cls_classes: int) -> None:
        self.cls_head = nn.Sequential(
            nn.Linear(dim_model, dim_model),  # pooler projection
            nn.Tanh(),                        # activation
            nn.Linear(dim_model, cls_classes) # classifier / regressor
        )


    def forward(self, gene_batch: torch.Tensor, expr_batch: torch.Tensor, with_context: bool = True) -> dict[str, torch.Tensor]:
        """
        gene_batch, expr_batch: (B, L) token ids from ``collate_fn``.

        Returns
        -------
        dict with
            'cls_embeddings'  : (B, D) CLS token output (default cell embedding)
            'mean_embeddings' : (B, D) masked mean over non-PAD, non-CLS tokens
                                (context tokens excluded when ``with_context=False``)
            'transformer_output': (B, L, D)
        """
        attention_mask = (gene_batch == PAD_TOKEN_ID)  # True for padding tokens

        gene_embedding = self.gene_embeddings(gene_batch)
        expression_embedding = self.expression_embeddings(expr_batch)

        # Create segment token
        segment_token = torch.zeros_like(gene_batch)  # cls token
        segment_token[:, 1:N_PREFIX_TOKENS] = 1  # context tokens
        segment_token[:, N_PREFIX_TOKENS:] = 2  # gene tokens
        segment_embedding = self.segment_embeddings(segment_token)

        embeddings = self.dropout(self.layer_norm(gene_embedding + expression_embedding + segment_embedding))

        embeddings = self.encoder(
            embeddings,
            src_key_padding_mask=attention_mask
        )

        cls_embeddings = embeddings[:, 0, :]

        # Base mask: exclude [PAD] and [CLS], True means valid token
        base_mask = ~(attention_mask | (gene_batch == CLS_TOKEN_ID))

        # Optionally exclude context tokens
        if not with_context:
            context_token_ids = torch.tensor(CONTEXT_GENE_TOKEN_IDS, device=gene_batch.device)
            base_mask = base_mask & ~torch.isin(gene_batch, context_token_ids)

        # Masked mean pooling
        mask_expanded = base_mask.unsqueeze(-1)  # (B, L, 1)
        sum_hidden = (embeddings * mask_expanded).sum(dim=1)
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1e-5)  # prevent divide-by-zero
        mean_embeddings = sum_hidden / valid_counts

        return {
            'cls_embeddings': cls_embeddings,
            'mean_embeddings': mean_embeddings,
            'transformer_output': embeddings,
        }


    @torch.no_grad()
    def extract_embs(self, batch, with_context: bool = True) -> dict[str, torch.Tensor]:
        """Get cell embeddings (no grad). ``batch`` is the tuple returned by ``collate_fn``."""
        gene_batch, expr_batch, *_ = batch
        return self.forward(gene_batch, expr_batch, with_context=with_context)


    def finetune(self, batch, with_context: bool = True) -> dict[str, torch.Tensor]:
        """Same as ``extract_embs`` but keeps the graph for fine-tuning."""
        gene_batch, expr_batch, *_ = batch
        return self.forward(gene_batch, expr_batch, with_context=with_context)


    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, map_location='cpu', **kwargs) -> 'GenVinci':
        """
        Load released weights from ``checkpoint_dir`` (``config.json`` + ``model.safetensors``).

        Extra ``kwargs`` override the config, e.g. ``cls_classes=10`` adds a freshly
        initialised ``cls_head`` for fine-tuning.
        """
        from safetensors.torch import load_file

        with open(os.path.join(checkpoint_dir, 'config.json')) as f:
            config = json.load(f)
        config.update(kwargs)

        model = cls(**config)
        state_dict = load_file(os.path.join(checkpoint_dir, 'model.safetensors'), device=str(map_location))

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith('cls_head.')]
        if missing or unexpected:
            raise RuntimeError(f'Checkpoint mismatch. missing: {missing}, unexpected: {unexpected}')
        return model
