import warnings
import numpy as np
import scanpy as sc
from scipy.sparse import issparse, csr_matrix

from .tokenizer.collator import gene_vocab, expr_vocab, CONTEXT_KEYS


def _check_raw_counts(X, n_check=1000):
    data = X[:n_check].data if issparse(X) else np.asarray(X[:n_check]).ravel()
    if data.size == 0:
        return
    if (data < 0).any():
        raise ValueError('adata.X contains negative values; GenVinci expects raw counts.')
    if not np.allclose(data, np.round(data)):
        warnings.warn('adata.X contains non-integer values; GenVinci expects raw counts '
                      '(normalize_total + log1p is applied internally).')


def load_data(adata, modality, organism, assay, tissue,
              gene_key=None, gene_selection='auto',
              n_top_genes=2000, hvg_threshold=2000, hvg_batch_key=None):
    """
    Map genes to the GenVinci vocabulary and normalize raw counts.

    Parameters
    ----------
    adata : AnnData with **raw counts** in ``adata.X``.
    modality, organism, assay, tissue : str
        Context metadata, see ``list_context_tokens()`` for valid values.
        Unknown values fall back to the 'UNKNOWN' token. Pass ``None`` to use an
        existing ``adata.obs`` column of the same name (per-cell values).
    gene_key : ``adata.var`` column with gene symbols / Ensembl IDs
        (default ``adata.var_names``). Names are matched exactly (case-sensitive,
        e.g. human 'GAPDH' and mouse 'Gapdh' are different tokens).
    gene_selection : 'auto' | 'top' | 'hvg'
        'top' : rank all genes of each cell by expression (no gene pre-selection).
        'hvg' : keep the ``n_top_genes`` highly variable genes first
                (``sc.pp.highly_variable_genes``, flavor='seurat', on log1p data).
        'auto': 'top' if the mean number of expressed genes per cell is below
                ``hvg_threshold``, otherwise 'hvg'.

    Returns
    -------
    AnnData (copy) with log1p-normalized ``X`` (CSR, float32),
    ``var['gene_token_id']``, ``var['raw_gene_names']`` and
    ``uns['genvinci']`` describing the preprocessing.
    """
    adata = adata.copy()

    _check_raw_counts(adata.X)

    # context metadata
    for key, value in zip(CONTEXT_KEYS, (modality, assay, organism, tissue)):
        if value is not None:
            adata.obs[key] = value
        elif key not in adata.obs:
            raise ValueError(f"'{key}' is None and adata.obs['{key}'] does not exist.")
        unknown = sorted(set(adata.obs[key].astype(str)) - set(expr_vocab))
        if unknown:
            warnings.warn(f"{key} value(s) {unknown} not in vocabulary, mapped to 'UNKNOWN'. "
                          f"See genvinci.cell_encoder.list_context_tokens().")

    # map genes to vocab ids
    gene_names = adata.var[gene_key].astype(str) if gene_key is not None else adata.var_names.astype(str)
    filter_genes = np.asarray(gene_names.isin(gene_vocab.keys()))
    print(f'Filtering out {adata.n_vars - filter_genes.sum()} genes not in vocab, remaining {filter_genes.sum()} genes.')
    if filter_genes.sum() == 0:
        raise ValueError('No gene matched the vocabulary. Use gene symbols or Ensembl IDs '
                         '(case-sensitive: human e.g. GAPDH, mouse e.g. Gapdh).')

    adata = adata[:, filter_genes].copy()
    adata.var['raw_gene_names'] = np.asarray(gene_names)[filter_genes]
    adata.var['gene_token_id'] = [gene_vocab[g] for g in adata.var['raw_gene_names']]

    # remove duplicated genes (several names mapped to the same token), keep the first
    adata = adata[:, ~adata.var.duplicated(subset='gene_token_id').values].copy()

    adata.X = csr_matrix(adata.X, dtype=np.float32) if not issparse(adata.X) else adata.X.tocsr().astype(np.float32)

    # decide gene selection
    mean_expressed_genes = float(np.mean(adata.X.getnnz(axis=1)))
    if gene_selection == 'auto':
        gene_selection = 'top' if mean_expressed_genes < hvg_threshold else 'hvg'
    if gene_selection not in ('top', 'hvg'):
        raise ValueError(f"gene_selection must be 'auto', 'top' or 'hvg', got {gene_selection!r}")
    print(f'Mean expressed genes per cell: {mean_expressed_genes:.0f} -> gene_selection = {gene_selection!r}')

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if gene_selection == 'hvg':
        n_top = min(n_top_genes, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor='seurat', batch_key=hvg_batch_key)
        adata = adata[:, adata.var['highly_variable'].values].copy()
        print(f'Kept {adata.n_vars} highly variable genes.')

    adata.X = adata.X.astype(np.float32)
    adata.var['gene_token_id'] = adata.var['gene_token_id'].astype(np.int64)
    adata.uns['genvinci'] = {
        'gene_selection': gene_selection,
        'mean_expressed_genes': mean_expressed_genes,
        'n_genes': int(adata.n_vars),
    }
    return adata


def list_context_tokens():
    """Valid values for the context metadata (modality / assay / organism / tissue)."""
    names = [k for k, v in sorted(expr_vocab.items(), key=lambda x: x[1]) if k not in ('CLS', 'PAD', 'MASK', 'UNKNOWN')]
    modality = ['single', 'spatial']
    organism = ['human', 'mouse']
    first_tissue = names.index('skin of body')
    assay = [n for n in names[:first_tissue] if n not in modality + organism]
    tissue = names[first_tissue:]
    return {'modality': modality, 'assay': assay, 'organism': organism, 'tissue': tissue}
