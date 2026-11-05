import os
import json
import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse
from torch.utils.data import Dataset

from .tools import binning, truncate


VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vocab')

with open(os.path.join(VOCAB_DIR, 'gene_vocab.json')) as f:
    gene_vocab = json.load(f)
with open(os.path.join(VOCAB_DIR, 'expr_vocab.json')) as f:
    expr_vocab = json.load(f)

UNK_EXPR_TOKEN_ID = expr_vocab['UNKNOWN']
BIN_TOKEN_OFFSET = max(expr_vocab.values()) + 1  # expression bin k -> token BIN_TOKEN_OFFSET + k

CONTEXT_KEYS = ('modality', 'assay', 'organism', 'tissue')


def tokenize_expr(expr_names, unk_token_id=UNK_EXPR_TOKEN_ID):
    return np.array([expr_vocab.get(expr, unk_token_id) for expr in expr_names], dtype=np.int64)


class CellDataset(Dataset):
    """Holds the preprocessed expression matrix; tokenisation happens in ``collate_fn``."""

    def __init__(self, X, gene_token_ids, context_token_ids, labels=None):
        self.X = csr_matrix(X, dtype=np.float32) if not issparse(X) else X.tocsr().astype(np.float32)
        self.gene_token_ids = np.asarray(gene_token_ids, dtype=np.int64)
        self.context_token_ids = np.asarray(context_token_ids, dtype=np.int64)  # (n_cells, 4)
        self.labels = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        input_dict = {
            'index': idx,
            'nonzero_values': row.data,
            'nonzero_gene_token_ids': self.gene_token_ids[row.indices],
            'context_token_ids': self.context_token_ids[idx],
        }
        if self.labels is not None:
            input_dict['labels'] = self.labels[idx]
        return input_dict


def adata_to_dataset(adata, label_key=None):
    """
    Wrap an AnnData returned by ``load_data`` into a ``CellDataset``.

    label_key: optional ``adata.obs`` column used as fine-tuning target.
    """
    if 'gene_token_id' not in adata.var:
        raise ValueError("adata.var['gene_token_id'] not found, run genvinci.cell_encoder.load_data first.")

    context = adata.obs[list(CONTEXT_KEYS)].astype(str).values
    context_token_ids = np.stack([tokenize_expr(row) for row in context])

    labels = adata.obs[label_key].values if label_key is not None else None

    return CellDataset(adata.X, adata.var['gene_token_id'].values, context_token_ids, labels=labels)


def collate_fn(batch, max_seq_len=1024, num_bins=50, pad_token_id=1, gene_selection='top', seed=0, labels=False):
    """
    Tokenise a batch of cells.

    Each cell's non-zero genes are binned into ``num_bins`` per-cell quantile bins,
    ranked by expression and truncated / padded to ``max_seq_len``.

    seed: randomness (tie-breaking, ``gene_selection='sampling'``) is seeded per cell
          by (seed, cell index), so results do not depend on batch size or workers.
          Set to None for non-deterministic behaviour.

    Returns (gene_token_ids, expr_token_ids, cell_indices[, labels])
    """
    gene_token_ids = []
    expr_token_ids = []
    batch_indexs = []
    batch_labels = []

    for input_dict in batch:
        rng = np.random.default_rng(None if seed is None else [seed, input_dict['index']])

        nonzero_values = np.asarray(input_dict['nonzero_values'], dtype=np.float64)
        nonzero_gene_token_ids = input_dict['nonzero_gene_token_ids']

        keep = nonzero_values > 0
        nonzero_values = nonzero_values[keep]
        nonzero_gene_token_ids = nonzero_gene_token_ids[keep]

        if len(nonzero_values) > 0:
            binned_nonzero_values = binning(nonzero_values, n_bins=num_bins + 1, rng=rng)  # 1 .. num_bins
            nonzero_expr_token_ids = binned_nonzero_values + BIN_TOKEN_OFFSET
        else:
            nonzero_expr_token_ids = np.zeros(0, dtype=np.int64)

        sampled_gene_token_ids, sampled_expr_token_ids = truncate(
            nonzero_values, nonzero_gene_token_ids, nonzero_expr_token_ids, input_dict['context_token_ids'],
            max_len=max_seq_len, pad_token_id=pad_token_id, gene_selection=gene_selection, rng=rng
        )
        gene_token_ids.append(torch.as_tensor(sampled_gene_token_ids, dtype=torch.long))
        expr_token_ids.append(torch.as_tensor(sampled_expr_token_ids, dtype=torch.long))
        batch_indexs.append(input_dict['index'])

        if labels:
            batch_labels.append(input_dict['labels'])

    batch = (
        torch.stack(gene_token_ids),
        torch.stack(expr_token_ids),
        torch.tensor(batch_indexs, dtype=torch.long),
    )
    if labels:
        batch = batch + (torch.tensor(np.asarray(batch_labels)),)

    return batch
