import numpy as np


def binning(non_zero_row, n_bins=51, rng=None):
    """
    Per-cell quantile binning of non-zero expression values.

    Values are mapped to ``1 .. n_bins - 1``; values that fall exactly on a bin
    edge are assigned to one of the two neighbouring bins at random.
    """
    rng = np.random.default_rng() if rng is None else rng

    def _digitize(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_digits = np.digitize(x, bins, right=True)

        rands = rng.random(len(x))  # uniform random numbers

        digits = rands * (right_digits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
    return _digitize(non_zero_row, bins)


def truncate(nonzero_values, nonzero_gene_token_ids, nonzero_expr_token_ids, context_token_ids,
             max_len=1024, pad_token_id=1, gene_selection='top', rng=None):
    """
    Rank genes by expression and keep ``max_len - 5`` of them, then prepend
    the CLS token and the 4 context tokens.

    gene_selection
        'top'      : keep the top ``max_len - 5`` expressed genes (default)
        'sampling' : sample ``max_len - 5`` genes without replacement,
                     with probability proportional to expression
    Cells with fewer genes are padded with ``pad_token_id``.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_keep = max_len - 5

    # shuffle first so that ties are broken at random
    shuffle_indices = rng.permutation(len(nonzero_gene_token_ids))
    nonzero_gene_token_ids = nonzero_gene_token_ids[shuffle_indices]
    nonzero_expr_token_ids = nonzero_expr_token_ids[shuffle_indices]
    nonzero_values = nonzero_values[shuffle_indices]

    # rank by expression, high to low
    ranked_indices = np.argsort(-nonzero_values, kind='stable')
    nonzero_gene_token_ids = nonzero_gene_token_ids[ranked_indices]
    nonzero_expr_token_ids = nonzero_expr_token_ids[ranked_indices]
    nonzero_values = nonzero_values[ranked_indices]

    # pad or truncate to max_len
    if len(nonzero_gene_token_ids) > n_keep:

        if gene_selection == 'top':
            sampled_gene_token_ids = nonzero_gene_token_ids[:n_keep]
            sampled_expr_token_ids = nonzero_expr_token_ids[:n_keep]

        elif gene_selection == 'sampling':
            probs = nonzero_values / np.sum(nonzero_values)
            sampled_indices = np.sort(rng.choice(len(nonzero_gene_token_ids), n_keep, replace=False, p=probs))
            sampled_gene_token_ids = nonzero_gene_token_ids[sampled_indices]
            sampled_expr_token_ids = nonzero_expr_token_ids[sampled_indices]

        else:
            raise ValueError(f"gene_selection must be 'top' or 'sampling', got {gene_selection!r}")

    else:  # pad with pad_token_id
        n_pad = n_keep - len(nonzero_gene_token_ids)
        sampled_gene_token_ids = np.pad(nonzero_gene_token_ids, (0, n_pad), 'constant', constant_values=pad_token_id)
        sampled_expr_token_ids = np.pad(nonzero_expr_token_ids, (0, n_pad), 'constant', constant_values=pad_token_id)

    # add fixed tokens: CLS + 4 context tokens
    sampled_gene_token_ids = np.concatenate([np.array([0, 3, 4, 5, 6]), sampled_gene_token_ids])
    sampled_expr_token_ids = np.concatenate([np.array([0]), np.asarray(context_token_ids), sampled_expr_token_ids])

    return sampled_gene_token_ids, sampled_expr_token_ids
