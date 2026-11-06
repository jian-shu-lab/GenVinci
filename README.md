# GenVinci

This repo includes a PyTorch implementation of **GenVinci**, from *GenVinci reconstructs 3D tissue scenes from histology to map cellular senescence across aging and disease*, with the following components:

- [x] **GenVinci cell encoder**: embeds single-cell and spatial transcriptomics data, and can be fine-tuned on cell-level tasks.
- [x] **GenVinci-2DST**: predicts single-cell gene expression and GenVinci cell embeddings from 2D H&E images.

GenVinci-3DST (3D tissue reconstruction from serial H&E sections) and GenVinci-gsMap (mapping human genetic associations onto reconstructed tissues) are in preparation.

## Installation

```
git clone https://github.com/jian-shu-lab/GenVinci.git
cd GenVinci
pip install -r requirements.txt
```

Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1B7dVpt6eUtL-SfCEhclmXz7xnQz6Ib7K?usp=sharing) and put them in `checkpoints/` as shown below. By downloading the weights, you agree to the [License and Terms of Use](#license-and-terms-of-use).

```
checkpoints/
├── cell_encoder/   config.json, model.safetensors
└── he2st/          config.json, model.safetensors
```

GenVinci-2DST uses [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) as the image feature extractor. It is a gated model: request access on its Hugging Face page, then log in with your own token (`huggingface-cli login`, or set `HF_TOKEN`).

## Embedding a new dataset

To generate an embedding for a new single-cell or spatial transcriptomics dataset in the AnnData format, use the `scripts/cell_encoder/get_cell_embs.py` script.

```
python scripts/cell_encoder/get_cell_embs.py --input {path_to_anndata} --modality {modality} --organism {organism} --assay {assay} --tissue {tissue}
```

where
- `input`: a h5ad file. The `.X` slot of the file should be **raw counts**. The `.var_names` slot should contain gene symbols or Ensembl IDs. Names are case-sensitive, so human `GAPDH` and mouse `Gapdh` are different genes.
- `modality`: `single` or `spatial`.
- `organism`: `human` or `mouse`.
- `assay`: e.g. `"10x 3' v3"`, `MERFISH`, `Xenium`.
- `tissue`: e.g. `"skin of body"`, `brain`.

Valid values for the last four arguments are listed by `genvinci.cell_encoder.list_context_tokens()`, and unknown values are mapped to `UNKNOWN`. If one of these arguments is omitted, the column with the same name in `adata.obs` is used instead.

Optional arguments:
- `gene_preselection`: `auto` (default), `top` or `hvg`. `hvg` keeps the top 2,000 highly variable genes (`sc.pp.highly_variable_genes`, `flavor="seurat"`). `auto` uses `hvg` when the mean number of expressed genes per cell is at least 2,000, and `top` otherwise.
- `gene_selection`: `top` (default) or `sampling`. Each cell is represented by at most 1,019 expressed genes. `top` keeps the most highly expressed genes, and `sampling` samples genes without replacement, with probability proportional to expression.
- `emb_type`: `cls` (default) or `mean`.
- `batch_size`: default 64.

The same pipeline can also be run from Python:

```python
import scanpy as sc
from genvinci.cell_encoder import GenVinci, load_data, get_cell_embs

adata = sc.read_h5ad("data.h5ad")
adata_tokens = load_data(adata, modality="single", organism="human", assay="10x 3' v3", tissue="skin of body")
model = GenVinci.from_pretrained("checkpoints/cell_encoder")
adata.obsm["X_genvinci"] = get_cell_embs(model, adata_tokens)
```

## Output

Final evaluated AnnData: `{input}_with_cell_embs.h5ad` (or `--output`). This AnnData is identical to the input AnnData, with GenVinci embeddings added in the `.obsm["X_genvinci"]` slot.

## Fine-tuning

To fine-tune GenVinci end-to-end on a cell-level classification or regression task, use the `scripts/cell_encoder/finetune.py` script.

```
python scripts/cell_encoder/finetune.py --input {path_to_anndata} --label_key {label_key} --task {task} --save_dir {save_dir} --modality {modality} --organism {organism} --assay {assay} --tissue {tissue}
```

where
- `label_key`: the column in `adata.obs` that holds the target.
- `task`: `classification` or `regression`.
- `save_dir`: the directory where the best checkpoint is saved. It can be reloaded with `GenVinci.from_pretrained(save_dir)`.
- `split_key` (optional): a column in `adata.obs` with values `train` / `val`. By default, 10% of the cells are randomly held out for validation.
- `no_standardize` (optional, regression only): by default the target is z-scored with the training mean and standard deviation, which are saved in `save_dir/finetune_info.json` (`prediction = output * label_std + label_mean`).

## Predicting expression from H&E (GenVinci-2DST)

To predict single-cell gene expression from an H&E image, use the `scripts/he2st/predict.py` script.

```
python scripts/he2st/predict.py --image {path_to_image} --mask {path_to_mask} --organism {organism} --tissue {tissue}
```

where
- `image`: an RGB H&E image (`.tif`, `.png`, `.jpg`) at **0.5 µm per pixel**. For images at another resolution, set `--mpp`.
- `mask`: a cell segmentation label mask (`.tif` or `.npy`) with the same height and width as the image. 0 is background and each cell has its own integer label. Alternatively, pass `--cells` with a csv containing `cell_id, x, y` (cell centres in image pixel coordinates). We do not provide segmentation code. Nuclei can be segmented with, for example, [StarDist](https://github.com/stardist/stardist) (`2D_versatile_he` model).
- `organism`: `human` or `mouse`.
- `tissue`: e.g. `"skin of body"`, `kidney`, `brain`.

Optional arguments:
- `assay` (default `Xenium`) and `condition` (default `normal`).
- `max_cells`: randomly use at most this many cells. The default is all cells.
- `batch_size`: default 256.

Valid metadata values are listed by `genvinci.he2st.list_metadata_tokens()`. For each cell, a 224 × 224 patch (112 µm) centred on the cell is used, and cells whose patch crosses the image border are skipped.

### Output

`{image}_pred.h5ad` (or `--output`): `.X` holds the predicted expression (cells × genes, with gene symbols of the given organism), `.obsm["spatial"]` holds the cell centres in pixels, and `.obsm["X_genvinci_pred"]` holds the predicted GenVinci cell embeddings.

## Data

The full ImageOmicsNet-300M dataset, including newly generated imaging and omics data and harmonized SenNet data, will be publicly available upon publication.

## Citing

```
@article{chen_genvinci,
  title = {GenVinci reconstructs 3D tissue scenes from histology to map cellular senescence across aging and disease},
  author = {Chen, Xingjian and others},
  year = {2026},
}
```

## License and Terms of Use

The GenVinci code and model weights are released under the [CC BY-NC-ND 4.0](LICENSE) license, which permits use only for non-commercial, academic research purposes with proper attribution. Non-commercial means "not primarily intended for or directed towards commercial advantage or monetary compensation" ([CC BY-NC-ND 4.0, Section 1(h)](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)).

By downloading or using the GenVinci models, you agree to the following terms:
- Any commercial use, sale or other monetization of the GenVinci models and their derivatives, which include models trained on outputs from the GenVinci models or datasets created from the GenVinci models, is prohibited and requires prior approval.
- You agree not to distribute, publish or reproduce a copy of the model weights. If another user within your organization wishes to use the GenVinci models, they must obtain the weights individually and agree to these terms.
- You may not attempt to re-identify the de-identified data used to develop the underlying models.

A US provisional patent application covering aspects of this work has been filed by Mass General Brigham, and no patent rights are granted by this license (see [NOTICE](NOTICE)). Commercial entities, including pharmaceutical and biotechnology companies, need a separate license: please contact Mass General Brigham Innovation or the corresponding author.

GenVinci-2DST uses [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1), which is released by Bioptimus under its own [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license and terms of use; by using GenVinci-2DST you also agree to its terms.
