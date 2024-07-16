import os, sys
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *
import wandb
import datetime
import argparse
from config import Config_Generative_Model
from dataset import create_spatial_dataset
from dc_ldm.ldm_for_rna import eLDM
from eval_metrics import fid_wrapper
import safetensors
from torch.utils.data import DataLoader

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def wandb_init(config):
    wandb.init( project="dreamdiffusion",
                group='eval',
                anonymous="allow",
                config=config,
                reinit=True)

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def get_args_parser():
    parser = argparse.ArgumentParser('Generating images from scRNA-Seq', add_help=False)
    
    # project parameters
    parser.add_argument('--dataset', type=str, default='morph_patchseq')
    parser.add_argument('--checkpoint', type=str, default='19-12-2023-02-42-55') # the checkpoint of saved SD
    
    # add an arg to operate extract embeddings or generation 
    parser.add_argument('--extract', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)

    # specify train or test
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    # batch size
    parser.add_argument('--batch_size', type=int, default=64)

    # specify the number of samples to generate
    parser.add_argument('--num_samples', type=int, default=1)

   
    return parser


if __name__ == '__main__':
    
    args = get_args_parser()
    args = args.parse_args()

    # args.generate = True
    # args.test = True
                                                                                   
    sd = torch.load(os.path.join('./output', args.dataset, 'ldm', args.checkpoint), map_location='cpu')
    config = sd['config'] # config.Config_Generative_Model
    
    # make sure config.dataset == args.dataset
    if config.dataset != args.dataset and args.dataset != 'scrna_seq':
        raise ValueError('Dataset and model mismatch')
    
    if args.extract:
        config.output_path = config.output_path.replace('ldm', 'extract') # has time stamp
    elif args.generate:
        config.output_path = config.output_path.replace('ldm', 'generate')
    else:
        raise ValueError('Please specify either extract or generate')
    
    os.makedirs(config.output_path, exist_ok=True)

    print(config.__dict__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    image_transform_train = transforms.Compose([ 
                    transforms.Resize((config.img_size, config.img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t*2.0)-1.0), # Scale between [-1, 1] 
                    channel_last # set channel last 
                    ])
                    
    image_transform_test = transforms.Compose([
                    transforms.Resize((config.img_size, config.img_size)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t*2.0)-1.0),
                    channel_last
                    ])

    gene_latents_dataset_train, gene_latents_dataset_val, gene_latents_dataset_test = create_spatial_dataset(config.dataset, config.tokenizer_path, config.test_ratio, image_transform=[image_transform_train, image_transform_test])

    # prepare pretrained conditional model              
    if config.pretrain_mlm_path is not None:                                  
        pretrain_mlm_metafile = safetensors.torch.load_file(os.path.join('output', config.dataset, 'mlm', config.pretrain_mlm_path, 'model.safetensors'), device='cpu')
                                # safetensors.torch.load_file(os.path.join(config.pretrain_mlm_path, 'model.safetensors'), device='cpu')
                                # torch.load(os.path.join(config.pretrain_mlm_path, 'pytorch_model.bin'), map_location='cpu')                        

    else: # use original pretrained geneformer
        pretrain_mlm_metafile = None 

    generative_model = eLDM(pretrain_mlm_metafile, ### None
                            device=device, 
                            pretrain_root=config.pretrain_ldm_path, # load config SD v1-5
                            logger=config.logger, # None
                            ddim_steps=config.ddim_steps, # 5
                            global_pool=config.global_pool, # False
                            use_time_cond=config.use_time_cond, # True
                            clip_tune=config.clip_tune, # True
                            cls_tune = config.cls_tune,
                            image_size = config.img_size,
                            ) # False
    
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=True)
    print('load ldm successfully')
    state = sd['state'] # cannot load for last but for previous best

    if args.train:
        gene_latents_dataset = gene_latents_dataset_train
        saved_grid_path = config.output_path + '/train_samples.png'
        config.output_path = config.output_path + '/train'
    elif args.test:
        gene_latents_dataset = gene_latents_dataset_test
        saved_grid_path = config.output_path + '/test_samples.png'
        config.output_path = config.output_path + '/test'

    if args.extract:
        gene_latents_dataloader = DataLoader(gene_latents_dataset, batch_size=args.batch_size, shuffle=False)
        gf_embs, text_embs, clip_embs = generative_model.get_embeddings(gene_latents_dataloader) 
        # save together into a dict to npy
        embeddings = {'gf_embs': gf_embs, 'text_embs': text_embs, 'clip_embs': clip_embs}
        np.save(config.output_path + '_embs.npy', embeddings)
    
    elif args.generate:
        # generate for each sample
        # os.makedirs(config.output_path, exist_ok=True)
        # grid, all_samples = generative_model.generate( # generate per sample
        #                                     gene_latents_dataset, 
        #                                     num_samples = num_samples, # 5
        #                                     ddim_steps = 250, # config.ddim_steps,
        #                                     HW = config.HW,
        #                                     limit = None,
        #                                     state=state,
        #                                     output_path = config.output_path,                         
        #                                     ) # num of prompts in test samples
        
        # # save the generated images
        # grid_imgs = Image.fromarray(grid.astype(np.uint8))
        # grid_imgs.save(saved_grid_path)

        # generate by rounds
        os.makedirs(config.output_path, exist_ok=True)
        if state is not None: # need to set first
            torch.cuda.set_rng_state(state)
        
        generated_samples = []
        # num_samples = 5 # sampling times
        for num in range(args.num_samples):
            grid, gt_samples, ddim_samples = generative_model.generate_per_batch(
                                            gene_latents_dataset, 
                                            args.batch_size,
                                            batch=num,
                                            ddim_steps = 250, # config.ddim_steps,
                                            HW = config.HW,
                                            limit = None,
                                            output_path = config.output_path,                         
                                            )
            generated_samples.append(ddim_samples)
        # combine gt and generated images
        
        generated_samples = np.concatenate(generated_samples, axis=2)
        all_samples = np.concatenate((gt_samples, generated_samples), axis=2)
        # sampling the first 8 images and save
        all_samples = all_samples[:8, :, :]
        all_samples = rearrange(all_samples, 'b c h w -> (b c) h w')
        Image.fromarray(all_samples).save(saved_grid_path)


