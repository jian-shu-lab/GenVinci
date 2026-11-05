"""
Extract GenVinci cell embeddings from raw counts.

Example
-------
python scripts/cell_encoder/get_cell_embs.py \
    --input data.h5ad --output data_with_cell_embs.h5ad \
    --checkpoint checkpoints/cell_encoder \
    --modality single --organism human --assay "10x 3' v3" --tissue "skin of body"
"""
import os
import sys
import argparse
import scanpy as sc

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genvinci.cell_encoder import GenVinci, load_data, get_cell_embs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='h5ad with raw counts in X')
    parser.add_argument('--output', default=None, help='default: <input>_with_cell_embs.h5ad')
    parser.add_argument('--checkpoint', default='checkpoints/cell_encoder', help='folder with config.json + model.safetensors')
    parser.add_argument('--modality', default=None, help="'single' or 'spatial' (None: use adata.obs['modality'])")
    parser.add_argument('--organism', default=None, help="'human' or 'mouse' (None: use adata.obs['organism'])")
    parser.add_argument('--assay', default=None, help="e.g. \"10x 3' v3\", 'MERFISH' (None: use adata.obs['assay'])")
    parser.add_argument('--tissue', default=None, help="e.g. 'skin of body', 'brain' (None: use adata.obs['tissue'])")
    parser.add_argument('--gene_key', default=None, help='adata.var column with gene names (default: var_names)')
    parser.add_argument('--gene_preselection', default='auto', choices=['auto', 'top', 'hvg'],
                        help="'auto': HVG if mean expressed genes per cell >= --hvg_threshold, else top")
    parser.add_argument('--n_top_genes', type=int, default=2000)
    parser.add_argument('--hvg_threshold', type=int, default=2000)
    parser.add_argument('--hvg_batch_key', default=None)
    parser.add_argument('--gene_selection', default='top', choices=['top', 'sampling'],
                        help='per-cell truncation when a cell has > max_seq_len - 5 expressed genes')
    parser.add_argument('--emb_type', default='cls', choices=['cls', 'mean'])
    parser.add_argument('--emb_key', default='X_genvinci')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    adata = sc.read_h5ad(args.input)

    adata_tokens = load_data(
        adata, modality=args.modality, organism=args.organism, assay=args.assay, tissue=args.tissue,
        gene_key=args.gene_key, gene_selection=args.gene_preselection,
        n_top_genes=args.n_top_genes, hvg_threshold=args.hvg_threshold, hvg_batch_key=args.hvg_batch_key,
    )

    genvinci = GenVinci.from_pretrained(args.checkpoint)

    adata.obsm[args.emb_key] = get_cell_embs(
        genvinci, adata_tokens, batch_size=args.batch_size, gene_selection=args.gene_selection,
        emb_type=args.emb_type, seed=args.seed, num_workers=args.num_workers, device=args.device,
    )
    adata.uns['genvinci'] = dict(adata_tokens.uns['genvinci'], gene_selection_per_cell=args.gene_selection,
                                 emb_type=args.emb_type, seed=args.seed)

    out_path = args.output or args.input.replace('.h5ad', '_with_cell_embs.h5ad')
    adata.write_h5ad(out_path)
    print(f'Saved embeddings to {out_path} (obsm["{args.emb_key}"], shape {adata.obsm[args.emb_key].shape})')


if __name__ == '__main__':
    main()
