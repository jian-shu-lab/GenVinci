import os
import json
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import anndata as ad
import tifffile
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


VOCAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab')
with open(os.path.join(VOCAB_DIR, 'metadata_vocab.json')) as f:
    META_VOCAB = json.load(f)
with open(os.path.join(VOCAB_DIR, 'gene_names.json')) as f:
    GENE_NAMES = json.load(f)

UNK_KEYS = {'assay': 'unknown_assay', 'organism': 'unknown_organism',
            'tissue': 'unknown_tissue', 'condition': 'unknown_condition'}

IMG_MODEL_NAME = 'hf-hub:bioptimus/H-optimus-1'
TARGET_MPP = 0.5
TARGET_SIZE = 224


# ============================================================
# Metadata
# ============================================================
def list_metadata_tokens():
    """Valid values for assay / organism / tissue / condition."""
    by_id = {v: k for k, v in META_VOCAB.items() if not k.startswith('unknown') and k != 'UNKNOWN'}
    return {
        'assay': [by_id[i] for i in range(0, 4)],
        'organism': [by_id[i] for i in range(4, 6)],
        'tissue': [by_id[i] for i in range(7, 41) if i in by_id],
        'condition': [by_id[i] for i in range(41, 44)],
    }


def get_metadata_ids(assay, organism, tissue, condition):
    valid = list_metadata_tokens()
    ids = {}
    for key, value in zip(('assay', 'organism', 'tissue', 'condition'), (assay, organism, tissue, condition)):
        if value in valid[key]:
            ids[key] = META_VOCAB[value]
        else:
            warnings.warn(f"{key}={value!r} not in vocabulary, using '{UNK_KEYS[key]}'. "
                          f"See genvinci.he2st.list_metadata_tokens().")
            ids[key] = META_VOCAB[UNK_KEYS[key]]
    return ids


# ============================================================
# Image / segmentation
# ============================================================
def load_image(path):
    """Load an RGB H&E image as uint8 (H, W, 3)."""
    if path.lower().endswith(('.tif', '.tiff')):
        image = tifffile.imread(path)
    else:
        Image.MAX_IMAGE_PIXELS = None
        image = np.array(Image.open(path).convert('RGB'))

    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)  # channel-first tif
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f'Expected an RGB image (H, W, 3), got shape {image.shape}')
    image = image[..., :3]
    if image.dtype != np.uint8:
        raise ValueError(f'Expected uint8 image, got {image.dtype}')
    return image


def load_cells_from_mask(mask_path):
    """
    Cell centres from a label mask (0 = background, k > 0 = cell k), e.g. the
    labels returned by StarDist. The centre of each cell's bounding box is used.
    """
    from skimage.measure import regionprops

    mask = np.load(mask_path) if mask_path.endswith('.npy') else tifffile.imread(mask_path)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f'Expected a 2D label mask, got shape {mask.shape}')

    cells = []
    for region in regionprops(mask.astype(np.int64)):
        y_min, x_min, y_max, x_max = region.bbox
        cells.append((int(region.label), (x_min + x_max) / 2, (y_min + y_max) / 2))
    return pd.DataFrame(cells, columns=['cell_id', 'x', 'y']), mask.shape


def load_cells_from_csv(csv_path):
    """Cell centres from a csv with columns ``cell_id, x, y`` (pixel coordinates)."""
    cells = pd.read_csv(csv_path)
    missing = {'cell_id', 'x', 'y'} - set(cells.columns)
    if missing:
        raise ValueError(f'{csv_path} is missing columns {sorted(missing)}')
    return cells[['cell_id', 'x', 'y']], None


def crop_patch_with_mpp(image, cx, cy, orig_mpp, target_mpp=TARGET_MPP, target_size=TARGET_SIZE):
    """Square patch centred on (cx, cy) covering target_size * target_mpp um; None at the border."""
    crop_size = int(round(target_size * target_mpp / orig_mpp))
    half = crop_size // 2

    H, W = image.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + crop_size, y1 + crop_size

    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None

    patch = image[y1:y2, x1:x2]
    if crop_size != target_size:
        patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return patch


# ============================================================
# Models
# ============================================================
def build_feature_extractor(device):
    """H-optimus-1 (gated on Hugging Face: accept the license and run `huggingface-cli login` first)."""
    import timm

    feature_extractor = timm.create_model(
        IMG_MODEL_NAME, pretrained=True, init_values=1e-5, dynamic_img_size=False,
    ).to(device).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.707223, 0.578729, 0.703617),
                             std=(0.211883, 0.230117, 0.177517)),
    ])
    return feature_extractor, transform


@torch.no_grad()
def extract_image_features(feature_extractor, x):
    """x: (B, 3, 224, 224) -> cls (B, 1536), patch tokens pooled to 8x8 (B, 64, 1536)."""
    tokens = feature_extractor.forward_features(x)
    B = x.shape[0]

    cls_emb = tokens[:, 0, :]

    patch_tok = tokens[:, 5:, :]   # skip cls + 4 register tokens -> 16x16 patches
    C = patch_tok.shape[-1]
    patch_tok = patch_tok.contiguous().view(B, 16, 16, C).permute(0, 3, 1, 2)
    patch_tok = torch.nn.functional.avg_pool2d(patch_tok, kernel_size=2, stride=2)
    patch_tok = patch_tok.permute(0, 2, 3, 1).reshape(B, 64, C)
    return cls_emb, patch_tok


# ============================================================
# Prediction
# ============================================================
@torch.no_grad()
def predict(image, cells, model, feature_extractor, transform,
            organism, tissue, assay='Xenium', condition='normal',
            mpp=TARGET_MPP, batch_size=256, max_cells=None, seed=42, device=None):
    """
    Predict single-cell gene expression from an H&E image.

    Parameters
    ----------
    image : uint8 RGB array (H, W, 3), ideally at 0.5 um / pixel.
    cells : DataFrame with columns cell_id, x, y (pixel coordinates in ``image``).
    organism : 'human' or 'mouse' (also selects the output gene names).
    tissue, assay, condition : see ``list_metadata_tokens()``.
    mpp : resolution of ``image`` in um / pixel.
    max_cells : randomly use at most this many cells (None = all cells).

    Returns
    -------
    AnnData: X = predicted expression (cells x genes),
             obsm['spatial'] = (x, y), obsm['X_genvinci_pred'] = predicted cell embedding.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if organism not in GENE_NAMES:
        raise ValueError(f"organism must be one of {list(GENE_NAMES)}, got {organism!r}")

    meta_ids = get_metadata_ids(assay, organism, tissue, condition)

    cells = cells.dropna(subset=['x', 'y']).reset_index(drop=True)
    n_total = len(cells)
    if max_cells and n_total > max_cells:
        rng = np.random.default_rng(seed)
        cells = cells.iloc[np.sort(rng.choice(n_total, size=max_cells, replace=False))].reset_index(drop=True)
    print(f'Using {len(cells)} / {n_total} cells')

    model = model.to(device).eval()
    feature_extractor = feature_extractor.to(device).eval()
    meta = {k: torch.tensor([v], device=device) for k, v in meta_ids.items()}

    all_keep, all_expr, all_emb = [], [], []

    def run_batch(patches):
        x = torch.stack([transform(p) for p in patches], dim=0).to(device)
        B = x.shape[0]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            cls_emb, patch_tok = extract_image_features(feature_extractor, x)
            outputs = model(patch_tok, cls_emb,
                            meta['assay'].expand(B), meta['organism'].expand(B),
                            meta['tissue'].expand(B), meta['condition'].expand(B))
        all_expr.append(outputs['pred_cell_expr'].float().cpu().numpy())
        all_emb.append(outputs['pred_cell_emb'].float().cpu().numpy())

    batch_patches = []
    for i, (cx, cy) in enumerate(tqdm(cells[['x', 'y']].values, total=len(cells))):
        patch = crop_patch_with_mpp(image, cx, cy, mpp)
        if patch is None:
            continue
        batch_patches.append(Image.fromarray(patch))
        all_keep.append(i)

        if len(batch_patches) >= batch_size:
            run_batch(batch_patches)
            batch_patches = []
    if batch_patches:
        run_batch(batch_patches)

    n_border = len(cells) - len(all_keep)
    if n_border:
        print(f'Skipped {n_border} cells whose patch crosses the image border')
    if not all_keep:
        raise ValueError('No usable cells.')

    # keep named genes of the organism
    names = np.array(GENE_NAMES[organism], dtype=object)
    gene_keep = np.array([n is not None for n in names])

    cells = cells.iloc[all_keep].reset_index(drop=True)
    adata = ad.AnnData(X=np.concatenate(all_expr)[:, gene_keep].astype(np.float32))
    adata.var_names = names[gene_keep].astype(str)
    adata.obs_names = cells['cell_id'].astype(str).values
    adata.obs['x'] = cells['x'].values
    adata.obs['y'] = cells['y'].values
    adata.obsm['spatial'] = cells[['x', 'y']].values.astype(np.float32)
    adata.obsm['X_genvinci_pred'] = np.concatenate(all_emb)
    adata.uns['genvinci_he2st'] = {
        'organism': organism, 'tissue': tissue, 'assay': assay, 'condition': condition,
        'mpp': mpp, 'n_cells_total': n_total, 'n_cells_used': int(adata.n_obs),
    }
    return adata
