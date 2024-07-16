import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
# from dc_ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from transformers import BertForSequenceClassification
import tqdm
import gc


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0


class classify_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(128, 1, 1, stride=1)#nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(1024, 40)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

class projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(2048, 1, 1, stride=1) # Pooling across the sequence length dimension
        self.fc = nn.Linear(256, 512)

    def forward(self, x): # [bs, 2048, 256]), no need to normalize for cosine similarity
        x = self.maxpool(x) # torch.Size([bs, 1, 256])
        x = x.squeeze(1) # torch.Size([bs, 256])
        x = self.fc(x) # torch.Size([bs, 512])
        return x

class cliprojection(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(77, 1, 1, stride=1) # Pooling across the sequence length dimension
        self.fc = nn.Linear(768, 512) # 768

    def forward(self, x): # [bs, 77, 768]), no need to normalize for cosine similarity
        x = self.maxpool(x) # torch.Size([bs, 1, 768])
        x = x.squeeze(1) # torch.Size([bs, 768])
        x = self.fc(x) # torch.Size([bs, 512])
        return x
   

class cond_stage_model_gene(nn.Module):
    def __init__(self, metafile, global_pool=True, clip_tune=True, cls_tune=False): # False, True, False
        super().__init__()

        # prepare pretrained geneformer
        model = BertForSequenceClassification.from_pretrained('ctheodoris/Geneformer',
                                                      num_labels=10,
                                                      output_attentions = False,
                                                      output_hidden_states = True)
        if metafile is not None:
            model.load_state_dict(metafile, strict=False) # replace to mlm pretrained geneformer

        self.mae = model
        if clip_tune: # project the geneformer output (bs, 2048, 256) to biomedCLIP image embedding (bs, 512)
            # self.mapping = projection() # for clip loss
            self.mapping = cliprojection()
        if cls_tune: # false    
            self.cls_net = classify_network()

        if global_pool == False: # # map geneformer output (bs, 2048, 256) to CLIP text output (bs, 77, 768)
            
            # self.channel_mapper = nn.Sequential(   
            #     nn.Conv1d(2048, 77, 1, stride=1),
            #     # nn.Conv1d(1024, 77, 1, bias=True)
            # )
            self.channel_mapper = nn.Sequential(   
                nn.Conv1d(2048, 77, 1, stride=1),
                nn.BatchNorm1d(77),  # Batch normalization for the output of the Conv1d layer
                nn.ReLU()  # ReLU activation function
                # nn.Conv1d(1024, 77, 1, bias=True)
            )
            
        # self.dim_mapper = nn.Linear(256, 768) # gene emb_dim
        self.dim_mapper = nn.Sequential(
        nn.Linear(256, 768),  # Linear layer mapping from 256 to 768
        nn.BatchNorm1d(77),  # Batch normalization for the output of the Linear layer
        nn.ReLU()  # ReLU activation function
    )


        self.global_pool = global_pool # False

        # self.loss_img = nn.CrossEntropyLoss()
        # self.loss_txt = nn.CrossEntropyLoss()

    def forward(self, x):

        # '<pad>': 0  # TODO: import attention mask      
        latent_crossattn = self.mae(x, attention_mask=(x!=0)).hidden_states[-1] # [bs, 2048] to [bs, 2048, 256]
        latent_return = latent_crossattn

        if self.global_pool == False: ### do this step
            latent_crossattn = self.channel_mapper(latent_crossattn) # torch.Size([bs, 2048, 256]) to torch.Size([bs, 77, 256])

        latent_crossattn = self.dim_mapper(latent_crossattn)   # torch.Size([bs, 77, 256]) to torch.Size([bs, 77, 768])
        out = latent_crossattn
        return out, latent_return  # torch.Size([bs, 77, 768]) and torch.Size([bs, 2048, 256])
    
    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        # image_embeds = self.image_embedder(image_inputs)
        target_emb = self.mapping(x) # (bs, 2048, 256) to (bs, 512)
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)

        target_emb = F.normalize(target_emb, dim=-1)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss
    

    def get_clip_ct_loss(self, x, image_embeds, logit_scale):
        # image_embeds = self.image_embedder(image_inputs)
        target_emb = self.mapping(x) # (bs, 2048, 256) to (bs, 512)
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)
        target_emb = F.normalize(target_emb, dim=-1)
        logits_per_image = logit_scale * image_embeds @ target_emb.t()
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(len(logits_per_image),dtype=torch.long).cuda()
        total_loss = (self.loss_img(logits_per_image,ground_truth) + self.loss_txt(logits_per_text,ground_truth))/2

        # # target_emb = F.normalize(target_emb, dim=-1)
        # loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return total_loss
    


class matcliprojection(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(77, 1, 1, stride=1) # Pooling across the sequence length dimension
        self.fc = nn.Linear(768, 512)

    def forward(self, x): # [bs, 77, 768]), no need to normalize for cosine similarity
        x = self.maxpool(x) # torch.Size([bs, 1, 768])
        x = x.squeeze(1) # torch.Size([bs, 768])
        x = self.fc(x) # torch.Size([bs, 512])
        return x


class cond_stage_model_mat(nn.Module):
    def __init__(self, global_pool=True, clip_tune=True, cls_tune=False): # False, True, False
        super().__init__()

        # # prepare pretrained geneformer
        # model = BertForSequenceClassification.from_pretrained('./pretrains/models/geneformer', # 'ctheodoris/Geneformer' 
        #                                               num_labels=10,
        #                                               output_attentions = False,
        #                                               output_hidden_states = True)
        # if metafile is not None:
        #     model.load_state_dict(metafile, strict=False) # replace to mlm pretrained geneformer

        # self.mae = model
        if clip_tune: # project the geneformer output (bs, 2048, 256) to biomedCLIP image embedding (bs, 512)
            # self.mapping = projection() # for clip loss
            self.mapping = matcliprojection()
        if cls_tune: # false    
            self.cls_net = classify_network()

        if global_pool == False: # # map geneformer output (bs, 2048, 256) to CLIP text output (bs, 77, 768)
            self.channel_mapper = nn.Sequential(   
                nn.Conv1d(512, 77, 1, bias=True),
                nn.Linear(256, 768)
            )
        self.dim_mapper = nn.Linear(256, 768, bias=True) # gene emb_dim
        self.global_pool = global_pool # False

    def forward(self, x):

        # '<pad>': 0  # TODO: import attention mask      
        # latent_crossattn = self.mae(x, attention_mask=(x!=0)).hidden_states[-1] # [bs, 2048] to [bs, 2048, 256]
        latent_return = x # 4, 512, 256

        if self.global_pool == False: ### do this step
            latent_crossattn = self.channel_mapper(x) # torch.Size([bs, 2048, 256]) to torch.Size([bs, 77, 256])
            # latent_crossattn = latent_crossattn.squeeze(1)
        # latent_crossattn = self.dim_mapper(latent_crossattn)   # torch.Size([bs, 77, 256]) to torch.Size([bs, 77, 768])
        out = latent_crossattn
        return out, latent_return  # torch.Size([bs, 77, 768]) and torch.Size([bs, 2048, 256])
    
    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        # image_embeds = self.image_embedder(image_inputs)
        target_emb = self.mapping(x) # (bs, 2048, 256) to (bs, 512)
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)

        # target_emb = F.normalize(target_emb, dim=-1)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss
    




class eLDM:

    def __init__(self, 
                 metafile=None, # pretrained conditional model # None
                 device=torch.device('cpu'),
                 pretrain_root=None, # '/home/chenxingjian/DreamDiffusion/pretrains'
                 logger=None, # None
                 ddim_steps=250, 
                 global_pool=True, # False
                 use_time_cond=False,  # True
                 clip_tune=True,  # True
                 cls_tune=False,
                 image_size=None): # False

        # prepare pretrained SD 
        self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt') # stable diffusion v1-5
        # https://huggingface.co/runwayml/stable-diffusion-v1-5
        # The Stable-Diffusion-v1-5 checkpoint was initialized with the weights of the Stable-Diffusion-v1-2 checkpoint and subsequently fine-tuned on 595k steps at resolution 512x512 on "laion-aesthetics v2 5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling.
        self.config_path = os.path.join(pretrain_root, 'models/v1-5-config.yaml') 
        config = OmegaConf.load(self.config_path) 


        # update image_size by input
        if image_size == 1024:
            config.model.params.image_size = 128
        elif image_size == 512:
            config.model.params.image_size = 64
        elif image_size == 256:
            config.model.params.image_size = 32
        elif image_size == 128:
            config.model.params.image_size = 16
        elif image_size == 64:
            config.model.params.image_size = 8


        config.model.params.unet_config.params.use_time_cond = use_time_cond # True, added
        config.model.params.unet_config.params.global_pool = global_pool    # False, added
        self.cond_dim = config.model.params.unet_config.params.context_dim  # 768

        # initialize SD model with the params in LatentDiffusion(DDPM)
        model = instantiate_from_config(config.model) # load 'dc_ldm.models.diffusion.ddpm.LatentDiffusion'

        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False) # load pretrained SD model
        
        # replace preset FrozenCLIPEmbedder with pretrained geneformer
        if hasattr(model, 'cond_stage_model'):
            del model.cond_stage_model
            gc.collect()  # Manually trigger Python garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache
            
        model.cond_stage_trainable = True ### added
        # replace preset FrozenCLIPEmbedder with pretrained geneformer
        model.cond_stage_model = cond_stage_model_gene(metafile, global_pool=global_pool, clip_tune=clip_tune, cls_tune=cls_tune)

        model.ddim_steps = ddim_steps # 
        model.re_init_ema()
        
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune # True
        self.model.cls_tune = cls_tune # False

        self.ldm_config = config
        self.pretrain_root = pretrain_root # '/home/chenxingjian/DreamDiffusion/pretrains'
        # self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        self.model.run_full_validation_threshold = 0.2

        print('\n##### Optimizing whole Stable Diffusion! #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True) # bs1: 4
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False) # revise bs1 to be 8
        self.model.unfreeze_whole_model() 
        self.model.freeze_first_stage()  # freeze AutoencoderKL
        self.model.freeze_image_embedder() # freeze BiomedCLIP
        self.model.freeze_cond_cls() # freeze cls layer of cond_stage_model

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = False # follow this one
        self.model.eval_avg = config.eval_avg # True
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader) 
        '''
        configure_optimizers, line 1554
        
        for train_batch in train_dataloader():
        on_train_batch_start, line 661, skip (not implemented)

        training_step, line 360
        on_train_batch_end

        validation_step, line 470
        '''
        
        # self.model.unfreeze_whole_model() # unfreeze to save the last model
        
        # torch.save(
        #     {
        #         'model_state_dict': self.model.state_dict(),
        #         'config': config,
        #         'state': torch.cuda.get_rng_state() # torch.random.get_rng_state()

        #     },
        #     os.path.join(output_path, 'checkpoint_last.pth') # final checkpoint
        # )


    def finetune_mat(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        self.model.run_full_validation_threshold = 0.2

        print('\n##### Optimizing whole Stable Diffusion! #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True) # bs1: 4
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False) # revise bs1 to be 8
        self.model.unfreeze_whole_model() 
        self.model.freeze_first_stage()  # freeze AutoencoderKL
        self.model.freeze_image_embedder() # freeze BiomedCLIP
        # self.model.freeze_cond_cls() # freeze cls layer of cond_stage_model

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = False # follow this one
        self.model.eval_avg = config.eval_avg # True
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader) 
        '''
        configure_optimizers, line 1554
        
        for train_batch in train_dataloader():
        on_train_batch_start, line 661, skip (not implemented)

        training_step, line 360
        on_train_batch_end

        validation_step, line 470
        '''
        
        self.model.unfreeze_whole_model() # unfreeze to save the last model
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.cuda.get_rng_state() # torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint_last.pth') # final checkpoint
        )





        

    @torch.no_grad()
    def generate(self, data_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path=None):
        # data_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels,  # latent shape
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(data_embedding): # set limit to test data or will have no stop
                if limit is not None: # for 10 samples
                    if count >= limit:
                        break

                # print(item)

                latent = item['count']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"Rendering {num_samples} examples in {ddim_steps} steps")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h -> c h', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put ground truth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, f'{count}-{copy_idx}.png'))
        
        # display as grid
        # sampling 20 images from all samples for visualization
        all_samples = all_samples[:20]

        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)




    @torch.no_grad()
    def generate_per_batch(self, dataset, batch_size, batch, ddim_steps, HW=None, limit=None, output_path = None):

        gt_samples = []
        ddim_samples = []

        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        # if state is not None:
        #     torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for count, item in enumerate(dataloader): # set limit to test data or will have no stop
                
                start_index = count * batch_size
                end_index = start_index + len(item['image'])
                indices = range(start_index, end_index)
                
                if limit is not None:
                    if count >= limit:
                        break

                # print(item)

                latent = item['count']
                gt_image = rearrange(item['image'], 'b h w c -> b c h w') # h w c
                print(f"Rendering {batch_size} examples in {ddim_steps} steps")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(latent.to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=len(c),
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                # all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=1)) # put groundtruth at first
                
                if output_path is not None:
                    x_samples_ddim = (255. * x_samples_ddim.detach().cpu().numpy()).astype(np.uint8)
                    gt_image = (255. * gt_image.detach().cpu().numpy()).astype(np.uint8)
                    
        
                for copy_idx, img_t, img_m in zip(indices, gt_image, x_samples_ddim):
                    img_t = rearrange(img_t, 'c h w -> h w c')
                    img_m = rearrange(img_m, 'c h w -> h w c')
                    
                    # save image
                    Image.fromarray(img_t).save(os.path.join(output_path, f'{copy_idx}_0.png'))
                    Image.fromarray(img_m).save(os.path.join(output_path, f'{copy_idx}_{batch + 1}.png'))

                    gt_samples.append(img_t)
                    ddim_samples.append(img_m)

                del x_samples_ddim, gt_image, samples_ddim, c, re_latent
                torch.cuda.empty_cache()
                gc.collect()



        gt_samples = np.stack(gt_samples, 0)
        ddim_samples = np.stack(ddim_samples, 0)

        # save for each round
        all_samples = np.concatenate([gt_samples, ddim_samples], axis=2)
        all_samples = rearrange(all_samples, 'b h w c ->  (b h) w c')
        # Image.fromarray(all_samples.astype(np.uint8)).save(os.path.join(output_path, f'round_{batch}.png'))

        return all_samples, gt_samples, ddim_samples


        # gt_samples
        # # display each round as grid
        # grid = torch.stack(all_samples, 0)
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        # grid = make_grid(grid, nrow=num_samples+1)

        # # to image
        # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        # model = model.to('cpu')
        
        # return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


    def generate_mat(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding): # set limit to test data or will have no stop
                if limit is not None:
                    if count >= limit:
                        break

                # print(item)

                latent = item['count']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"Rendering {num_samples} examples in {ddim_steps} steps")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                # gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)



    @torch.no_grad()
    def get_embeddings(self, dataloader):

        model = self.model.to(self.device)

        # freeze the whole model
        model.freeze_whole_model()

        gf_embs = []
        text_embs = []
        clip_embs = []
  
        with model.ema_scope():
            model.eval()
            for step, item in tqdm.tqdm(enumerate(dataloader)): 
                latent = item['count']
                c, re_latent = model.get_learned_conditioning(latent.to(self.device))

                text_embs.append(c.mean(1).cpu().numpy())
                gf_embs.append(re_latent.mean(1).cpu().numpy())

                clip_latent = model.cond_stage_model.mapping(c)
                # print(c.shape, re_latent.shape) # c is the text embedding 
                clip_embs.append(clip_latent.cpu().numpy())

        gf_embs = np.concatenate(gf_embs, axis=0)
        text_embs = np.concatenate(text_embs, axis=0)
        clip_embs = np.concatenate(clip_embs, axis=0)

        return gf_embs, text_embs, clip_embs


    @torch.no_grad()
    def get_zero_emb(self, dataset):
        model = self.model.to(self.device)
        with model.ema_scope():
            model.eval()
            latent = dataset['count']
            c, re_latent = model.get_learned_conditioning(latent.unsqueeze(0).to(self.device))
        return c[0].cpu().numpy()
