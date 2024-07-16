import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy
import safetensors
import torch.distributed as dist

###
from config import Config_Generative_Model 
from dataset import  create_spatial_dataset
from dc_ldm.ldm_for_rna import eLDM
from eval_metrics import get_similarity_metric


def wandb_init(config, output_path, group=None):

    wandb.login()

    wandb.init( project='genvinci',
                group=group,
                anonymous="allow",
                config=config,
                reinit=True)
    
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    # metric_list = ['mse', 'pcc', 'ssim', 'psm']
    metric_list = ['fid']
    res_list = []
    
    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            # res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res = get_similarity_metric(pred_images, gt_images, method='metrics-only', metric_name='fid')
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))     
    # res_part = []
    # for s in samples_to_run:
    #     pred_images = [img[s] for img in samples]
    #     pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
    #     res = get_similarity_metric(pred_images, gt_images, 'class', None, 
    #                     n_way=50, num_trials=50, top_k=1, device='cuda')
    #     res_part.append(np.mean(res))
    # res_list.append(np.mean(res_part))
    # res_list.append(np.max(res_part))
    # metric_list.append('top-1-class')
    # metric_list.append('top-1-class (max)')
    return res_list, metric_list
               
def generate_images(generative_model, rna_latents_dataset_train, rna_latents_dataset_val, config):
   
    # generate for all the training samples
    grid, _ = generative_model.generate(rna_latents_dataset_train, config.num_samples, # line 376
                config.ddim_steps, config.HW, 8) # HW: None, here limit is for all the train/val samples: 10

    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})
    
    # generate for all the testing samples
    grid, samples = generative_model.generate(rna_latents_dataset_val, config.num_samples, 
                config.ddim_steps, config.HW, 8) 
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_val.png'))
    wandb.log({f'summary/samples_val': wandb.Image(grid_imgs)})

    # for sp_idx, imgs in enumerate(samples): # for each samples
    #     for copy_idx, img in enumerate(imgs[1:]):
    #         img = rearrange(img, 'c h w -> h w c')
    #         Image.fromarray(img).save(os.path.join(config.output_path, 
    #                         f'./val{sp_idx}-{copy_idx}.png'))

    # metric, metric_list = get_eval_metric(samples)
    # metric_dict = {f'summary/metrics-only_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    # metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    # metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    # print(metric_dict)

    # wandb.log(metric_dict)

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def update_config(args, config):
    '''
    overwrite the attributes of a config object with the values from an args object, 
    only for those attributes that exist in both objects and have a non-None value in the args object.
    '''
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def create_trainer(num_epoch, precision=16, accumulate_grad_batches=None,logger=None,check_val_every_n_epoch=None):
    # num_epoch = 500
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger, 
            precision=precision, accumulate_grad_batches=accumulate_grad_batches, 
            log_every_n_steps=1,
            
            # devices=1, # for single gpu
            
            devices=2, 
            num_nodes=1,
            strategy="ddp", 

            enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5, num_sanity_val_steps=0, # revise for bigger dataset?
            check_val_every_n_epoch=check_val_every_n_epoch) # every 1 epoch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def scale(t):
            return (t*2.0)-1.0

def get_args_parser():
    parser = argparse.ArgumentParser('Conditioning LDM Finetuning', add_help=False)
   
    # project parameters
    parser.add_argument('--seed', type=int)

    ### need to be specified
    parser.add_argument('--dataset', type=str, default='he_st') 
    # datasets = ['he_merfish/cell_snrna', 'he_merfish/tile_scrna', 'he_st', 'he_visium', 'dapi_cosmx/nucleus_snrna', 'dapi_cosmx/cell_scrna', 'dapi_cosmx/composite_scrna', 'em_merfish', 'morph_patchseq']
   
    parser.add_argument('--pretrain_mlm_path', type=str, default='240104_230849') ### for scRNA-seq pretrained model
    parser.add_argument('--img_size', type=int, default=256) # 256

    parser.add_argument('--checkpoint_path', type=str) # for resume training
    parser.add_argument('--pretrain_ldm_path', type=str) # for stable diffusion

    # finetune parameters
    parser.add_argument('--batch_size', type=int, default=1) # default=4
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int, default=101) # default=100
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)
    parser.add_argument('--use_time_cond', type=bool)

    # diffusion sampling parameters
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int, default=250) # 250
    parser.add_argument('--eval_avg', type=bool)

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser

def main(config):
    
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    image_transform_train = transforms.Compose([ 
                    transforms.Resize((config.img_size, config.img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                    transforms.ToTensor(),
                    # AddGaussianNoise(0., 0.001),  # Add Gaussian Noise with a standard deviation of 0.05
                    transforms.Lambda(scale), # Scale between [-1, 1] 
                    channel_last # set channel last 
                    ])
                    
    image_transform_test = transforms.Compose([
                    transforms.Resize((config.img_size, config.img_size)),
                    transforms.ToTensor(),
                    transforms.Lambda(scale),
                    channel_last
                    ])
    
    gene_latents_dataset_train, gene_latents_dataset_val, gene_latents_dataset_test = create_spatial_dataset(config.dataset, config.tokenizer_path, config.test_ratio, image_transform=[image_transform_train, image_transform_test])
    # else: # TODO: for other datasets

    # prepare pretrained conditional model              
    if config.pretrain_mlm_path is not None:                                  
        pretrain_mlm_metafile = safetensors.torch.load_file(os.path.join('output', config.dataset, 'mlm', config.pretrain_mlm_path, 'model.safetensors'), device='cpu')
                                # safetensors.torch.load_file(os.path.join(config.pretrain_mlm_path, 'model.safetensors'), device='cpu')
                                # torch.load(os.path.join(config.pretrain_mlm_path, 'pytorch_model.bin'), map_location='cpu')                        

    else: # use original pretrained geneformer
        pretrain_mlm_metafile = None 

    # create generateive model
    generative_model = eLDM(pretrain_mlm_metafile, ### None
                            device=device, 
                            pretrain_root=config.pretrain_ldm_path, # './pretrains'
                            logger=config.logger, # None
                            ddim_steps=config.ddim_steps, # 250
                            global_pool=config.global_pool, # False
                            use_time_cond=config.use_time_cond, # True
                            clip_tune=config.clip_tune, # True
                            cls_tune = config.cls_tune,
                            image_size = config.img_size,
                            ) # False
    
    # resume training if applicable
    if config.checkpoint_path is not None: # skip
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('Model resumed')

    # finetune the model
    trainer = create_trainer(config.num_epoch, config.precision, config.accumulate_grad, config.logger, check_val_every_n_epoch=1)
    generative_model.finetune(trainer, gene_latents_dataset_train, gene_latents_dataset_val,
                config.batch_size, config.lr, config.output_path, config=config) 

    # generate limited train and test images and generate images for subjects seperately
    generate_images(generative_model, gene_latents_dataset_train, gene_latents_dataset_test, config) # gene_latents_dataset_val

    print('\n##### Finished! #####')

    wandb.finish()

    return

  
if __name__ == '__main__':

    args = get_args_parser()   # set defalut params
    args = args.parse_args()

    config = Config_Generative_Model() # load params from config.py
    config = update_config(args, config) # update config using the params from args only for those attributes that exist in both objects and have a non-None value in the args object
    
    if config.checkpoint_path is not None: # resume training
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        checkpoint_path_meta = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = checkpoint_path_meta
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    output_path = os.path.join(config.output_path, config.dataset, 'ldm',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path

    os.makedirs(output_path, exist_ok=True)
    wandb_init(config, output_path, group=config.dataset)

    logger = WandbLogger()
    config.logger = logger # logger
    main(config)


