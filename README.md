# Generation of cellular and tissue images from single-cell expression profiles with GenVinci
<p align="center">
<img src="overview.png" width="80%" height="80%">
</p>

## GenVinci
**GenVinci** is a generative model that learns a multi-modal representation of morphological and molecular profiles and uses it for gene-to-image translation across various profiling platforms and imaging techniques. GenVinci’s cascaded architecture allows us to flexibly apply it across a range of downstream tasks, including image generation, multi-modal clustering, zero-shot classification, and data augmentation. This document introduces the procedures required for replicating the results in *Generation of cellular and tissue images from single-cell expression profiles with GenVinci*.

## Abstract
Cells were discovered and originally characterized through imaging hundreds of years ago, whereas recent advances in single-cell genomics are enabling their comprehensive cataloging by comprehensive molecular profiles. However, most technologies characterize cells by one modality at a time, requiring substantial measurement effort to collect multiple views and leading to multiple, seemingly disparate views of their identity. This is largely because the mapping between views is non-linear and not well-understood. Moreover, while imaging data at the cell and tissue level has longstanding interpretation in clinical practice through histopathology, molecular profiles are more recent, mostly destructive, and much more costly. Recent advances in generative AI and deep learning have opened the way to generate multiple views through non-linear transformation and may offer a way to unify different aspects of cells and tissues. Here, we introduce GenVinci, a generative model based on transformer and diffusion models that learns a multi-modal representation of cells and tissues and reconstructs morphological information from transcriptome profiles to generate cellular and tissue images from single-cell expression profiles during inference. We demonstrate how GenVinci can generate diverse types of imaging data, including histological (hematoxylin and eosin (H&E)) stains, fluorescence microscopy images, neuron morphology, and electron microscopy, from various molecular profiles at different resolutions. GenVinci generates accurate cellular and tissue images compared to ground-truth morphologies, cell types, and pathological annotations. It outperforms a convolutional neural network (CNN) baseline in both generation quality and semantic mapping. Thanks to its cascaded architecture, GenVinci can be flexibly applied to a range of downstream tasks, including image generation, multi-modal clustering, zero-shot classification, and data augmentation. GenVinci can unify different views of cell and tissue biology, while greatly reducing the need for multiple measurements, towards an ultimate goal of virtual cell and tissue simulators.

## Datasets
We analyzed six spatial transcriptomics datasets across different sequencing resolutions and technologies (Spatial Transcriptomics, 10x Genomics Visium, HTAPP MERFISH and scRNA-seq, Nanostring CosMx SMI, Patch-seq, and STcEM MERFISH). The download links can be found in the manuscript.

## Installation and Usage

Please install and utilize conda (https://docs.anaconda.com/miniconda/) to create and activate a environment.
```bash
$ git clone https://github.com/jian-shu-lab/GenVinci.git
$ cd GenVinci
$ conda env create -f env.yaml
$ conda activate genvinci
```

## Continued pre-training
The continued pre-training step is based on Geneformer (https://huggingface.co/ctheodoris/Geneformer) and we used it to further pre-train our own spatial transcriptomics datasets. Before running the following scripts, please download and move the Geneformer folder (https://huggingface.co/ctheodoris/Geneformer/tree/main/geneformer) into the 'src/pretrain' folder and preprocess your dataset as Geneformer requested (https://huggingface.co/ctheodoris/Geneformer/tree/main/examples).

```bash
# please specify your dataset at the end of the command, here we use 'he_st' dataset as an example
deepspeed ./src/pretrain/pretrain_geneformer_w_deepspeed.py --deepspeed ds_config.json he_st
```

## Fine tuning using Stable Diffusion
We fine-tuned a Stable Diffusion model (https://github.com/CompVis/stable-diffusion) for gene expression-to-image generation. For Stable Diffusion, we just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You need the file ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).
```bash
python3 ./src/finetune/rna_ldm.py --dataset he_st --pretrain_mlm_path 240104_230849 --img_size 256
```

## Generation of cellular images from gene expression with trained checkpoints
The trained GenVinci is used to generate cellular images from gene expression profiles.
```bash
python3 ./src/finetune/gen_from_rna.py --dataset he_st --checkpoint checkpoint_100.pth --generate --test --num_samples 5 --batch_size 4
```

## Checkpoint and demo dataset
We provide a checkpoint trained on one of Spatial Transcriptomics datasets and H&E images for test. The generated images like following (the first column is the ground truth and we randomly generate 5 times from the same gene expression profiles).
<p align="center">
<img src="./output/he_st/generate/09-01-2024-00-08-28/he_st/generate/12-01-2024-15-46-25/test_samples.png" width="50%" height="50%">
</p>


## Manuscript
Please refer to our manuscript Xingjian et al. (2024) for more details.


## Acknowledgement

This code is built upon the publicly available code [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [Mind-vis](https://github.com/zjc062/mind-vis) and [StableDiffusion](https://github.com/CompVis/stable-diffusion). Thanks these authors for making their excellent work and codes publicly available.



Contact
-----------------------
We are happy about any feedback! If you have any questions, please feel free to contact Xingjian Chen (xchen57@mgh.harvard.edu). Find more research in SHU LAB (https://www.jianshulab.org/).

