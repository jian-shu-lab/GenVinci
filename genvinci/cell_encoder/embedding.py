from functools import partial

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from .tokenizer import collate_fn, adata_to_dataset


def get_cell_embs(model, adata, batch_size=64, gene_selection='top', emb_type='cls',
                  seed=0, num_workers=4, device=None):
    """
    Extract cell embeddings.

    Parameters
    ----------
    model : GenVinci
    adata : AnnData returned by ``load_data``.
    gene_selection : 'top' (default) or 'sampling'
        How genes are chosen when a cell has more than ``max_seq_len - 5``
        expressed genes: top expressed, or sampled without replacement with
        probability proportional to expression.
    emb_type : 'cls' (default) or 'mean'
    seed : per-cell random seed for tie-breaking / sampling (None = not seeded).

    Returns
    -------
    np.ndarray (n_cells, dim_model)
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = adata_to_dataset(adata)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_seq_len=model.config['max_seq_len'], num_bins=model.config['num_bins'],
                           gene_selection=gene_selection, seed=seed),
        num_workers=num_workers,
        pin_memory=True,
    )

    model = model.to(device)
    model.eval()
    all_embs = []
    for batch in tqdm.tqdm(dataloader):
        batch_tokens = tuple(t.to(device) for t in batch)
        embs = model.extract_embs(batch_tokens)[f'{emb_type}_embeddings']
        all_embs.append(embs.float().cpu())

    return torch.cat(all_embs, dim=0).numpy()
