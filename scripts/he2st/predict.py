"""
Predict single-cell gene expression from an H&E image with GenVinci-HE2ST.

Example
-------
python scripts/he2st/predict.py \
    --image slide_0p5mpp.tif --mask slide_0p5mpp_labels.tif \
    --organism human --tissue "skin of body" --output slide_pred.h5ad
"""
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genvinci.he2st import (HE2ST, predict, load_image, load_cells_from_mask, load_cells_from_csv,
                             build_feature_extractor)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', required=True, help='RGB H&E image (tif / png / jpg), ideally 0.5 um per pixel')
    seg = parser.add_mutually_exclusive_group(required=True)
    seg.add_argument('--mask', help='cell label mask (.tif / .npy), same size as --image, 0 = background')
    seg.add_argument('--cells', help='csv with columns cell_id, x, y (pixel coordinates in --image)')
    parser.add_argument('--organism', required=True, choices=['human', 'mouse'])
    parser.add_argument('--tissue', required=True, help="e.g. 'skin of body', 'brain'; see list_metadata_tokens()")
    parser.add_argument('--assay', default='Xenium')
    parser.add_argument('--condition', default='normal')
    parser.add_argument('--mpp', type=float, default=0.5, help='um per pixel of --image')
    parser.add_argument('--output', default=None, help='default: <image>_pred.h5ad')
    parser.add_argument('--checkpoint', default='checkpoints/he2st')
    parser.add_argument('--hf_token', default=None,
                        help='Hugging Face token with access to bioptimus/H-optimus-1 '
                             '(default: HF_TOKEN env variable or `huggingface-cli login`)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_cells', type=int, default=None, help='randomly use at most this many cells')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    image = load_image(args.image)
    print(f'Image: {image.shape}')

    if args.mask:
        cells, mask_shape = load_cells_from_mask(args.mask)
        if mask_shape != image.shape[:2]:
            raise ValueError(f'Mask shape {mask_shape} does not match image shape {image.shape[:2]}')
    else:
        cells, _ = load_cells_from_csv(args.cells)
    print(f'Cells: {len(cells)}')

    model = HE2ST.from_pretrained(args.checkpoint)
    feature_extractor, transform = build_feature_extractor(args.device or 'cpu')

    adata = predict(
        image, cells, model, feature_extractor, transform,
        organism=args.organism, tissue=args.tissue, assay=args.assay, condition=args.condition,
        mpp=args.mpp, batch_size=args.batch_size, max_cells=args.max_cells, seed=args.seed, device=args.device,
    )

    out_path = args.output or os.path.splitext(args.image)[0] + '_pred.h5ad'
    adata.write_h5ad(out_path)
    print(f'Saved {adata.shape} predictions to {out_path}')


if __name__ == '__main__':
    main()
