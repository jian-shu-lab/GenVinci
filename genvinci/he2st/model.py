import os
import json

import torch
import torch.nn as nn


class SpecialEmbedding(nn.Module):
    """Shared metadata embedding table (assay / species / organ / condition)."""
    def __init__(self, num_embeddings=64, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, token_ids):
        """
        token_ids: (B,)
        returns:   (B, D)
        """
        return self.embedding(token_ids)


class HE2ST(nn.Module):
    """
    Predicts gene expression and a GenVinci cell embedding from H-optimus-1
    features of a 224 x 224 patch (0.5 mpp) centred on a cell.
    """
    def __init__(
        self,
        img_dim=1536,
        hidden_dim=512,
        panel_size=25030,
        emb_dim=512,
        num_special_embeddings=64,
        **kwargs
    ):
        super().__init__()

        self.config = dict(img_dim=img_dim, hidden_dim=hidden_dim, panel_size=panel_size,
                           emb_dim=emb_dim, num_special_embeddings=num_special_embeddings)

        # shared special embedding table
        self.special_emb = SpecialEmbedding(num_special_embeddings, hidden_dim)

        # separate projections for each metadata type
        self.assay_proj = nn.Linear(hidden_dim, hidden_dim)
        self.species_proj = nn.Linear(hidden_dim, hidden_dim)
        self.organ_proj = nn.Linear(hidden_dim, hidden_dim)
        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)

        # separate scales
        self.assay_scale = nn.Parameter(torch.tensor(0.0))
        self.species_scale = nn.Parameter(torch.tensor(0.0))
        self.organ_scale = nn.Parameter(torch.tensor(0.0))
        self.condition_scale = nn.Parameter(torch.tensor(0.0))
        self.cls_scale = nn.Parameter(torch.tensor(0.0))

        # per-bin initialization
        self.bin_init = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # local bin interaction on 8x8 grid
        self.local_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # expression head
        self.bin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, panel_size),
            nn.Softplus(),
        )

        # cell embedding head
        self.emb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        # cls projection
        self.cls_proj = nn.Linear(img_dim, hidden_dim)

    def forward(
        self,
        patch_tokens,
        cls_emb,
        assay_id,
        species_id,
        organ_id,
        condition_id,
    ):
        """
        patch_tokens: (B, 64, img_dim)  8x8 pooled patch tokens
        cls_emb:      (B, img_dim)
        assay_id:     (B,)
        species_id:   (B,)
        organ_id:     (B,)
        condition_id: (B,)
        """
        B, N, D = patch_tokens.shape
        assert N == 64, f"Expected 64 tokens, got {N}"
        assert D == self.config['img_dim'], f"Expected img_dim={self.config['img_dim']}, got {D}"

        # 1) pure image bin embedding
        bin_latent_img = self.bin_init(patch_tokens)   # (B, 64, H)

        # 2) metadata embeddings
        special_emb = (
            self.assay_scale * self.assay_proj(self.special_emb(assay_id))
            + self.species_scale * self.species_proj(self.special_emb(species_id))
            + self.organ_scale * self.organ_proj(self.special_emb(organ_id))
            + self.condition_scale * self.condition_proj(self.special_emb(condition_id))
        )  # (B, H)

        # 3) cls global information
        cls_latent = self.cls_scale * self.cls_proj(cls_emb)   # (B, H)

        # 4) inject metadata + cls into every bin
        bin_latent_0 = bin_latent_img + special_emb.unsqueeze(1) + cls_latent.unsqueeze(1)   # (B, 64, H)

        # 5) local refine
        H = bin_latent_0.shape[-1]
        bin_grid = bin_latent_0.view(B, 8, 8, H).permute(0, 3, 1, 2)   # (B, H, 8, 8)
        bin_grid = self.local_refine(bin_grid)                          # (B, H, 8, 8)
        bin_latent_1 = bin_grid.permute(0, 2, 3, 1).reshape(B, 64, H)  # (B, 64, H)

        # 6) expression and embedding prediction per bin
        pred_bin_expr = self.bin_head(bin_latent_1)    # (B, 64, panel_size)
        pred_bin_emb = self.emb_head(bin_latent_1)     # (B, 64, emb_dim)

        # 7) cell-level prediction: mean of the central 2x2 bins
        pred_cell_expr = pred_bin_expr.view(B, 8, 8, -1)[:, 3:5, 3:5, :].mean(dim=(1, 2))   # (B, panel_size)
        pred_cell_emb = pred_bin_emb.view(B, 8, 8, -1)[:, 3:5, 3:5, :].mean(dim=(1, 2))     # (B, emb_dim)

        return {
            "pred_bin_expr": pred_bin_expr,
            "pred_bin_emb": pred_bin_emb,
            "pred_cell_expr": pred_cell_expr,
            "pred_cell_emb": pred_cell_emb,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_dir, map_location='cpu'):
        """Load released weights from ``checkpoint_dir`` (``config.json`` + ``model.safetensors``)."""
        from safetensors.torch import load_file

        with open(os.path.join(checkpoint_dir, 'config.json')) as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(load_file(os.path.join(checkpoint_dir, 'model.safetensors'), device=str(map_location)))
        return model.eval()
