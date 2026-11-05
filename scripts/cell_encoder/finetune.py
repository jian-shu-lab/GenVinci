"""
Fine-tune GenVinci end-to-end on a cell-level classification or regression task.

Example
-------
python scripts/cell_encoder/finetune.py \
    --input data.h5ad --label_key cell_type --task classification \
    --checkpoint checkpoints/cell_encoder --save_dir finetuned/cell_type \
    --modality single --organism human --assay "10x 3' v3" --tissue "skin of body"
"""
import os
import sys
import json
import copy
import argparse
from functools import partial

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from genvinci.cell_encoder import GenVinci, load_data, collate_fn, adata_to_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='h5ad with raw counts in X')
    parser.add_argument('--label_key', required=True, help='adata.obs column with the target')
    parser.add_argument('--task', default='classification', choices=['classification', 'regression'])
    parser.add_argument('--split_key', default=None,
                        help="adata.obs column with 'train' / 'val' / 'test'; default: random --val_size split")
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--no_standardize', action='store_true',
                        help='regression only: do not z-score the target (default: standardize with train mean / std)')
    parser.add_argument('--checkpoint', default='checkpoints/cell_encoder')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--modality', default=None)
    parser.add_argument('--organism', default=None)
    parser.add_argument('--assay', default=None)
    parser.add_argument('--tissue', default=None)
    parser.add_argument('--gene_key', default=None)
    parser.add_argument('--gene_preselection', default='auto', choices=['auto', 'top', 'hvg'])
    parser.add_argument('--n_top_genes', type=int, default=2000)
    parser.add_argument('--hvg_threshold', type=int, default=2000)
    parser.add_argument('--gene_selection', default='top', choices=['top', 'sampling'])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def run_epoch(genvinci, dataloader, loss_fn, device, task, optimizer=None):
    training = optimizer is not None
    genvinci.train(training)

    total_loss, all_preds, all_labels, all_indexs = 0, [], [], []
    with torch.set_grad_enabled(training):
        for batch in dataloader:
            batch_tokens = tuple(t.to(device) for t in batch)
            cls_emb = genvinci.finetune(batch_tokens)['cls_embeddings']
            pred_labels = genvinci.cls_head(cls_emb)
            real_labels = batch_tokens[3]
            if task == 'regression':
                real_labels = real_labels.float().view(-1, 1)
            loss = loss_fn(pred_labels, real_labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.append(pred_labels.detach().cpu())
            all_labels.append(real_labels.cpu())
            all_indexs.append(batch_tokens[2].cpu())

    return total_loss / len(dataloader), torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_indexs)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    adata = adata[~adata.obs[args.label_key].isna()].copy()

    # encode labels
    if args.task == 'classification':
        classes = sorted(adata.obs[args.label_key].astype(str).unique())
        class_to_id = {c: i for i, c in enumerate(classes)}
        adata.obs['_genvinci_label'] = adata.obs[args.label_key].astype(str).map(class_to_id).astype(np.int64)
        cls_classes = len(classes)
    else:
        classes = None
        adata.obs['_genvinci_label'] = adata.obs[args.label_key].astype(np.float32)
        cls_classes = 1

    # split
    if args.split_key is not None:
        split = adata.obs[args.split_key].astype(str).values
        train_idx, val_idx = np.where(split == 'train')[0], np.where(split == 'val')[0]
    else:
        stratify = adata.obs['_genvinci_label'].values if args.task == 'classification' else None
        train_idx, val_idx = train_test_split(np.arange(adata.n_obs), test_size=args.val_size,
                                              random_state=args.seed, stratify=stratify)
    print(f'train: {len(train_idx)} cells, val: {len(val_idx)} cells')

    # standardize regression target with train statistics
    label_scaler = None
    if args.task == 'regression' and not args.no_standardize:
        labels = adata.obs['_genvinci_label'].values.reshape(-1, 1)
        label_scaler = StandardScaler().fit(labels[train_idx])
        adata.obs['_genvinci_label'] = label_scaler.transform(labels).ravel().astype(np.float32)
        print(f'Standardized {args.label_key}: mean {label_scaler.mean_[0]:.4g}, std {label_scaler.scale_[0]:.4g}')

    adata_tokens = load_data(
        adata, modality=args.modality, organism=args.organism, assay=args.assay, tissue=args.tissue,
        gene_key=args.gene_key, gene_selection=args.gene_preselection,
        n_top_genes=args.n_top_genes, hvg_threshold=args.hvg_threshold,
    )
    dataset = adata_to_dataset(adata_tokens, label_key='_genvinci_label')

    genvinci = GenVinci.from_pretrained(args.checkpoint, cls_classes=cls_classes).to(args.device)

    collate = partial(collate_fn, max_seq_len=genvinci.config['max_seq_len'], num_bins=genvinci.config['num_bins'],
                      gene_selection=args.gene_selection, seed=args.seed, labels=True)
    train_dataloader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(genvinci.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss() if args.task == 'classification' else torch.nn.MSELoss()

    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0

    for epoch in range(args.epochs):
        train_loss, *_ = run_epoch(genvinci, train_dataloader, loss_fn, args.device, args.task, optimizer)
        scheduler.step()
        val_loss, val_preds, val_labels, _ = run_epoch(genvinci, val_dataloader, loss_fn, args.device, args.task)

        msg = f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}'
        if args.task == 'classification':
            msg += f', Val Acc: {(val_preds.argmax(1) == val_labels).float().mean():.4f}'
        print(msg)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(genvinci.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print('Early stopping triggered.')
                break

    # save best model
    genvinci.load_state_dict(best_model_state)
    save_file({k: v.contiguous() for k, v in best_model_state.items()}, os.path.join(args.save_dir, 'model.safetensors'))
    config = {k: v for k, v in genvinci.config.items() if k != 'dropout'}
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.save_dir, 'finetune_info.json'), 'w') as f:
        json.dump({'task': args.task, 'label_key': args.label_key, 'classes': classes,
                   # prediction in original units = output * label_std + label_mean
                   'label_mean': float(label_scaler.mean_[0]) if label_scaler is not None else None,
                   'label_std': float(label_scaler.scale_[0]) if label_scaler is not None else None,
                   'best_val_loss': best_val_loss, **adata_tokens.uns['genvinci']}, f, indent=2)
    print(f'Saved fine-tuned model to {args.save_dir}')


if __name__ == '__main__':
    main()
