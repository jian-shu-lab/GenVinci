# Generation of cellular and tissue images from single-cell expression profiles with GenVinci
<p align="center">
<img src="overview.png" width="70%" height="70%">
</p>

## GenVinci
**GenVinci** is a generative model that learns a multi-modal representation of morphological and molecular profiles and uses it for gene-to-image translation across various profiling platforms and imaging techniques. GenVinci’s cascaded architecture allows us to flexibly apply it across a range of downstream tasks, including image generation, multi-modal clustering, zero-shot classification, and data augmentation. This document introduces the procedures required for replicating the results in *Generation of cellular and tissue images from single-cell expression profiles with GenVinci*.

## Abstract
Cells were discovered and originally characterized through imaging hundreds of years ago, whereas recent advances in single-cell genomics are enabling their comprehensive cataloging by comprehensive molecular profiles. However, most technologies characterize cells by one modality at a time, requiring substantial measurement effort to collect multiple views and leading to multiple, seemingly disparate views of their identity. This is largely because the mapping between views is non-linear and not well-understood. Moreover, while imaging data at the cell and tissue level has longstanding interpretation in clinical practice through histopathology, molecular profiles are more recent, mostly destructive, and much more costly. Recent advances in generative AI and deep learning have opened the way to generate multiple views through non-linear transformation and may offer a way to unify different aspects of cells and tissues. Here, we introduce GenVinci, a generative model based on transformer and diffusion models that learns a multi-modal representation of cells and tissues and reconstructs morphological information from transcriptome profiles to generate cellular and tissue images from single-cell expression profiles during inference. We demonstrate how GenVinci can generate diverse types of imaging data, including histological (hematoxylin and eosin (H&E)) stains, fluorescence microscopy images, neuron morphology, and electron microscopy, from various molecular profiles at different resolutions. GenVinci generates accurate cellular and tissue images compared to ground-truth morphologies, cell types, and pathological annotations. It outperforms a convolutional neural network (CNN) baseline in both generation quality and semantic mapping. Thanks to its cascaded architecture, GenVinci can be flexibly applied to a range of downstream tasks, including image generation, multi-modal clustering, zero-shot classification, and data augmentation. GenVinci can unify different views of cell and tissue biology, while greatly reducing the need for multiple measurements, towards an ultimate goal of virtual cell and tissue simulators.

## Continued pre-training
The continued pre-training step is based on Geneformer and we used it to further pre-train our own spatial transcriptomics datasets.

## Fine tuning using Stable Diffusion
We fine-tuned a Stable Diffusion model for gene expression-to-image generation.

## Manuscript
Please refer to our manuscript Xingjian et al. (2024) for more details.

Contact
-----------------------
We are happy about any feedback! If you have any questions, please feel free to contact Xingjian Chen (xchen57@mgh.harvard.edu).
Find more research in SHU LAB (https://www.jianshulab.org/).

