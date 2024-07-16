"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities import rank_zero_only

from dc_ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from dc_ldm.modules.ema import LitEma
from dc_ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from dc_ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from dc_ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from dc_ldm.models.diffusion.ddim import DDIMSampler
from dc_ldm.models.diffusion.plms import PLMSSampler
from PIL import Image
import torch.nn.functional as F
from eval_metrics import get_similarity_metric
import open_clip
from torchmetrics.image.fid import FrechetInceptionDistance

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

from dc_ldm.modules.encoders.modules import FrozenImageEmbedder
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ddim_steps=300
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)  # wrap here, unet config
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.validation_count = 0
        self.ddim_steps = ddim_steps
        self.return_cond = False
        self.output_path = None
        self.main_config = None

        self.best_val = 1e10
        self.state = None

        self.run_full_validation_threshold = 0.0
        self.eval_avg = True

        self.validation_step_gt_samples = []
        self.validation_step_ddim_samples = []


    def re_init_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}")   

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO: how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema: # false
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long() # 1000 steps
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key) 
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        self.train()
        self.cond_stage_model.train()
        
        loss, loss_dict = self.shared_step(batch) # line 1064
        # Note: loss on progress bar is a running average
        # train/loss_simple, train/loss_vlb, train/loss, train/loss_clip
        self.log_dict(loss_dict, prog_bar=True, # should adjust x-axis in wandb
                    logger=True, on_step=True, on_epoch=True) ### logger=True
        # self.log("global_step", self.global_step,
        #          prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # self.logger.log_metrics(loss_dict, step=self.global_step)

        if self.use_scheduler: # skip
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    # TODO:
    # def training_epoch_end(self):
    # def on_train_epoch_end(self):
    #     all_preds = torch.stack(self.training_step_outputs)
    #     # do something with all preds
    #     ...
    #     self.training_step_outputs.clear()  # free memory

    @torch.no_grad()
    def generate(self, data, num_samples, ddim_steps=300, HW=None, limit=None, state=None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None: ### (4, 64, 64)
            shape = (self.p_channels, 
                self.p_image_size, self.p_image_size)
        else:
            num_resolutions = len(self.ch_mult)
            shape = (self.p_channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self
        sampler = PLMSSampler(model)  # full validation
        # sampler = DDIMSampler(model)
        model.eval()
        if torch.cuda.is_available():
            state = torch.cuda.get_rng_state() if state is None else state
            torch.cuda.set_rng_state(state) # Returns the random number generator state
        else:
            state = torch.get_rng_state() if state is None else state
            torch.set_rng_state(state)

        # rng = torch.Generator(device=self.device).manual_seed(2022).set_state(state)

        # state = torch.cuda.get_rng_state()    
        with model.ema_scope():
             
            for count, item in enumerate(zip(data['count'], data['image'])): # for each prompt in the batch
                if limit is not None: # generate how many images for each prompt
                    if count >= limit: # limit the number of prompt to sample in each batch
                        break
                latent = item[0] # fmri embedding
                gt_image = rearrange(item[1], 'h w c -> 1 c h w') # h w c
                print(f"Rendering {num_samples} examples in {ddim_steps} steps") # rendering 5 examples in 250 steps
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # [bs, 77, 768] and [bs, 2048, 256]

                if len(latent.shape) == 1: # for tile level
                    c, re_latent = model.get_learned_conditioning(repeat(latent, 'h -> c h', c=num_samples).to(self.device))
                else: # for slide level
                    c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))  # repeat num_samples times
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,                 # reduce the noise
                                                conditioning=c, # [bs, 77, 768]
                                                batch_size=num_samples,
                                                shape=shape, # (4, 64, 64)
                                                verbose=False,
                                                generator=None) 
                # torch.Size([5, 4, 64, 64])
                x_samples_ddim = model.decode_first_stage(samples_ddim) # recover the image
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0,min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image.detach().cpu(), x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
        
        # display as grid
        grid = torch.stack(all_samples, 0) # [bs, gt + num_samples, 4, 64, 64]
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy() # big image
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8), state



    @torch.no_grad()
    def generate_per_batch(self, data, ddim_steps=300, HW=None, state=None):

        if HW is None: ### (4, 64, 64)
            shape = (self.p_channels, 
                self.p_image_size, self.p_image_size)
        else:
            num_resolutions = len(self.ch_mult)
            shape = (self.p_channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self
        sampler = PLMSSampler(model)  # full validation
        # sampler = DDIMSampler(model)
        model.eval()
        if torch.cuda.is_available():
            state = torch.cuda.get_rng_state() if state is None else state
            torch.cuda.set_rng_state(state) # Returns the random number generator state
        else:
            state = torch.get_rng_state() if state is None else state
            torch.set_rng_state(state)

        # rng = torch.Generator(device=self.device).manual_seed(2022).set_state(state)
        # state = torch.cuda.get_rng_state()    
        
        with model.ema_scope():
            
            latent = data['count']
            gt_image = rearrange(data['image'], 'b h w c -> b c h w') # h w c
            print(f"Rendering {len(latent)} examples in {ddim_steps} steps") # rendering 5 examples in 250 steps
            c, re_latent = model.get_learned_conditioning(latent.to(self.device)) 
            samples_ddim, _ = sampler.sample(S=ddim_steps,                 # reduce the noise
                                            conditioning=c, # [bs, 77, 768]
                                            batch_size=len(latent),
                                            shape=shape, # (4, 64, 64)
                                            verbose=False,
                                            generator=None) 
            
            x_samples_ddim = model.decode_first_stage(samples_ddim) # recover the image
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0).detach().cpu()
            gt_image = torch.clamp((gt_image+1.0)/2.0,min=0.0, max=1.0).detach().cpu()

            # gt_samples = gt_image.detach().cpu()
            # ddim__samples = x_samples_ddim.detach().cpu()
            # all_samples.append(torch.cat([gt_image.detach().cpu(), x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
    
        #  # display as grid
        # grid = torch.stack(all_samples, 0) # [bs, gt + num_samples, 4, 64, 64]
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        # grid = make_grid(grid, nrow=x_samples_ddim.shape[0]+1)

        # # to image
        # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy() # big image
        # return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8), (255. * gt_samples.numpy()).astype(np.uint8), (255. * ddim__samples.numpy()).astype(np.uint8), state
        return (255. * gt_image).to(torch.uint8), (255. * x_samples_ddim).to(torch.uint8)



    def generate_mat(self, data, num_samples, ddim_steps=300, HW=None, limit=None, state=None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None: ### (4, 64, 64)
            shape = (self.p_channels, 
                self.p_image_size, self.p_image_size)
        else:
            num_resolutions = len(self.ch_mult)
            shape = (self.p_channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self
        sampler = PLMSSampler(model)  # full validation
        # sampler = DDIMSampler(model)
        model.eval()
        if torch.cuda.is_available():
            state = torch.cuda.get_rng_state() if state is None else state
            torch.cuda.set_rng_state(state) # Returns the random number generator state
        else:
            state = torch.get_rng_state() if state is None else state
            torch.set_rng_state(state)

        # rng = torch.Generator(device=self.device).manual_seed(2022).set_state(state)

        # state = torch.cuda.get_rng_state()    
        with model.ema_scope():
            for count, item in enumerate(zip(data['count'], data['image'])): # for each prompt in the batch
                if limit is not None: # generate how many images for each prompt
                    if count >= limit: # limit the number of prompt to sample in each batch
                        break
                latent = item[0] # fmri embedding
                gt_image = rearrange(item[1], 'h w c -> 1 c h w') # h w c
                print(f"Rendering {num_samples} examples in {ddim_steps} steps") # rendering 5 examples in 250 steps
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # [bs, 77, 768] and [bs, 2048, 256]
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))  # repeat num_samples times
                samples_ddim, _ = sampler.sample(S=ddim_steps,                 # reduce the noise
                                                conditioning=c, # [bs, 77, 768]
                                                batch_size=num_samples,
                                                shape=shape, # (4, 64, 64)
                                                verbose=False,
                                                generator=None) 
                # torch.Size([5, 4, 64, 64])
                x_samples_ddim = model.decode_first_stage(samples_ddim) # recover the image
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0,min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image.detach().cpu(), x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
        
        # display as grid
        grid = torch.stack(all_samples, 0) # [bs, gt + num_samples, 4, 64, 64]
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy() # big image
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8), state




    def save_images(self, all_samples, suffix=0):
        # print('output_path')
        # print(self.output_path)
        if self.output_path is not None:
            os.makedirs(os.path.join(self.output_path, 'val', f'{self.current_epoch}_{suffix}'), exist_ok=True)
            for sp_idx, imgs in enumerate(all_samples):
                # for copy_idx, img in enumerate(imgs[1:]):
                for copy_idx, img in enumerate(imgs):
                    img = rearrange(img, 'c h w -> h w c')
                    Image.fromarray(img).save(os.path.join(self.output_path, 'val', 
                                    f'{self.current_epoch}_{suffix}', f'val_{sp_idx}-{copy_idx}.png'))
                                    
    # def full_validation(self, batch, state=None): # for each batch
    #     print('###### Run full validation! ######\n')
    #     # grid, all_samples, state = self.generate_mat(batch, ddim_steps=self.ddim_steps, num_samples=5, limit=None, state=state)
    #     gt_samples, ddim_simples, state = self.generate(batch, ddim_steps=self.ddim_steps, state=state)
    #     return gt_samples, ddim_simples, state 
        
        # # here run PLMSSampler, all_samples: (bs, gt+num_samples, 3, 512, 512)
        # metric, metric_list = self.get_eval_metric(all_samples)
        # self.save_images(all_samples, suffix='%.4f'%metric[-1])
        # metric_dict = {f'val/{k}_first':v for k, v in zip(metric_list, metric)}
        # # print(metric_dict) # fid for each epoch first val batch
        # # self.logger.log_metrics(metric_dict, step=self.current_epoch)
        # self.log_dict(metric_dict, prog_bar=False, # should adjust x-axis in wandb
        #             logger=True, on_step=False, on_epoch=True) ### logger=True
        # grid_imgs = Image.fromarray(grid.astype(np.uint8))
        # self.logger.log_image(key=f'samples_val_first', images=[grid_imgs], step=self.current_epoch) # save images
        
        # # if metric[-1] > self.best_val: # 0
        # #     self.best_val = metric[-1]
        # if self.current_epoch == 0 or (self.current_epoch+1) % 5 == 0: # save every 5 epochs?
        #     torch.save(
        #         {
        #             'model_state_dict': self.state_dict(),
        #             'config': self.main_config,
        #             'state': state

        #         },
        #         os.path.join(self.output_path, 'checkpoint_{}.pth'.format(self.current_epoch)) # save every epoch
        #     )

    def full_validation(self, batch, state=None): # for each batch
        print('###### Run full validation! ######\n')
        # grid, all_samples, state = self.generate_mat(batch, ddim_steps=self.ddim_steps, num_samples=5, limit=None, state=state)
        # grid, all_samples, state = self.generate(batch, ddim_steps=self.ddim_steps, num_samples=5, limit=None, state=state)
        # # here run PLMSSampler, all_samples: (bs, gt+num_samples, 3, 512, 512)
        # metric, metric_list = self.get_eval_metric(all_samples)
        # self.save_images(all_samples, suffix='%.4f'%metric[-1])
        # metric_dict = {f'val/{k}_first':v for k, v in zip(metric_list, metric)}
        # print(metric_dict) # fid for each epoch first val batch
        # self.logger.log_metrics(metric_dict, step=self.current_epoch)
        # self.log_dict(metric_dict, prog_bar=False, # should adjust x-axis in wandb
        #             logger=True, on_step=False, on_epoch=True) ### logger=True
        # grid_imgs = Image.fromarray(grid.astype(np.uint8))
        # self.logger.log_image(key=f'samples_val_first', images=[grid_imgs], step=self.current_epoch) # save images
        
        # # if metric[-1] > self.best_val: # 0
        # #     self.best_val = metric[-1]
        # if self.current_epoch == 0 or (self.current_epoch+1) % 5 == 0: # save every 5 epochs?
        #     torch.save(
        #         {
        #             'model_state_dict': self.state_dict(),
        #             'config': self.main_config,
        #             'state': state

        #         },
        #         os.path.join(self.output_path, 'checkpoint_{}.pth'.format(self.current_epoch)) # save every epoch
        #     )

    # @torch.no_grad()
    # def on_validation_epoch_end(self): 


    #     if self.current_epoch != 0 and self.current_epoch % 20 == 0:
    #         print('\n###### Calculate the FID for full validation! ######\n')
            
    #         gt_samples = torch.cat(self.validation_step_gt_samples, axis = 0)
    #         ddim_simples = torch.cat(self.validation_step_ddim_samples, axis = 0)
            
    #         # calculate fid
    #         fid = FrechetInceptionDistance(feature=2048)
    #         fid.update(gt_samples, real=True)
    #         fid.update(ddim_simples, real=False)
    #         val_fid = fid.compute().item()
    #         fid.reset()

    #         metric_dict = {f'val/fid_all':val_fid}
    #         self.log_dict(metric_dict, prog_bar=False, # should adjust x-axis in wandb
    #                     logger=True, on_step=False, on_epoch=True) ### logger=True

    #         if val_fid < self.best_val: # 1e10
    #             self.best_val = val_fid
    #             print('###### Best FID is {} at epoch {}, saving checkpoint! ######\n'.format(self.best_val, self.current_epoch))
    #             torch.save(
    #                     {
    #                         'model_state_dict': self.state_dict(),
    #                         'config': self.main_config,
    #                         'state': self.state

    #                     },
    #                     os.path.join(self.output_path, 'checkpoint_best.pth'.format(self.current_epoch)) # save every epoch
    #                 )
    #         else:
    #             print('###### Current FID is {}, best FID is {} at epoch {} ######\n'.format(val_fid, self.best_val, self.current_epoch))


    #         self.validation_step_gt_samples.clear()  # free memory
    #         self.validation_step_ddim_samples.clear()  # free memory
            
            # del gt_samples
            # del ddim_simples


        # self.save_images(all_samples, suffix='%.4f'%metric[-1])
        # metric_dict = {f'val/{k}_first':v for k, v in zip(metric_list, metric)}
        # # print(metric_dict) # fid for each epoch first val batch
        # # self.logger.log_metrics(metric_dict, step=self.current_epoch)
        # self.log_dict(metric_dict, prog_bar=False, # should adjust x-axis in wandb
        #             logger=True, on_step=False, on_epoch=True) ### logger=True
        # grid_imgs = Image.fromarray(grid.astype(np.uint8))
        # self.logger.log_image(key=f'samples_val_first', images=[grid_imgs], step=self.current_epoch) # save images
        
        # # if metric[-1] > self.best_val: # 0
        # #     self.best_val = metric[-1]
        # if self.current_epoch == 0 or (self.current_epoch+1) % 5 == 0: # save every 5 epochs?
        #     torch.save(
        #         {
        #             'model_state_dict': self.state_dict(),
        #             'config': self.main_config,
        #             'state': state

        #         },
        #         os.path.join(self.output_path, 'checkpoint_{}.pth'.format(self.current_epoch)) # save every epoch
        #     )



    @torch.no_grad()
    def validation_step(self, batch, batch_idx):  ### run this step after each epoch
        self.eval()
        self.cond_stage_model.eval()

        # print the val loss
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch) # do nothing just predict again since ema = False
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)



        if batch_idx == 0 and self.current_epoch % 20 == 0: 
            
            print('\n ###### Run partial validation! ######\n')                                   # 5 and 8
            grid, all_samples, state = self.generate(batch, ddim_steps=self.ddim_steps, num_samples=1, limit=1, state=None) # only show the first epoch
            # here run PLMSSampler, all_samples: (bs, gt+num_samples, 3, 512, 512)
            # metric, metric_list = self.get_eval_metric(all_samples)
            self.save_images(all_samples, suffix='0')
            # metric_dict = {f'val/{k}_first':v for k, v in zip(metric_list, metric)}
            # print(metric_dict) # fid for each epoch first val batch
            # self.logger.log_metrics(metric_dict, step=self.current_epoch)
            # self.log_dict(metric_dict, prog_bar=False, # should adjust x-axis in wandb
            #             logger=True, on_step=False, on_epoch=True) ### logger=True
            grid_imgs = Image.fromarray(grid.astype(np.uint8))
            self.logger.log_image(key=f'samples_val_first', images=[grid_imgs], step=self.current_epoch) # save images
            self.state = state

            # only save on one process
            if self.trainer.is_global_zero and self.current_epoch % 50 == 0: # save every 10 epochs?
                    torch.save(
                        {
                            'model_state_dict': self.state_dict(),
                            'config': self.main_config,
                            'state': state

                        },
                        os.path.join(self.output_path, 'checkpoint_{}.pth'.format(self.current_epoch)) 
                    )


        # if batch_idx == 0: 
            
        #     print('\n ###### Run partial validation! ######\n')                                   # 5 and 8
        #     grid, all_samples, state = self.generate(batch, ddim_steps=self.ddim_steps, num_samples=5, limit=2, state=None) # only show the first epoch
        #     # here run PLMSSampler, all_samples: (bs, gt+num_samples, 3, 512, 512)
        #     metric, metric_list = self.get_eval_metric(all_samples)
        #     self.save_images(all_samples, suffix='%.4f'%metric[-1])
        #     metric_dict = {f'val/{k}_first':v for k, v in zip(metric_list, metric)}
        #     # print(metric_dict) # fid for each epoch first val batch
        #     self.logger.log_metrics(metric_dict, step=self.current_epoch)
        #     self.log_dict(metric_dict, prog_bar=False, # should adjust x-axis in wandb
        #                 logger=True, on_step=False, on_epoch=True) ### logger=True
        #     grid_imgs = Image.fromarray(grid.astype(np.uint8))
        #     self.logger.log_image(key=f'samples_val_first', images=[grid_imgs], step=self.current_epoch) # save images
        #     self.state = state
        #     # self.validation_count += 1

        #     # # if metric[-1] > self.best_val: # 0
        #     # #     self.best_val = metric[-1]
        #     if self.current_epoch % 10 == 0: # save every 10 epochs?
        #         torch.save(
        #             {
        #                 'model_state_dict': self.state_dict(),
        #                 'config': self.main_config,
        #                 'state': state

        #             },
        #             os.path.join(self.output_path, 'checkpoint_{}.pth'.format(self.current_epoch)) 
        #         )
            
        #         # print('\n###### Run full validation! ######\n')
        #         # gt_samples, ddim_simples = self.generate_per_batch(batch, ddim_steps=self.ddim_steps, HW=None, state=self.state)
        #         # self.validation_step_gt_samples.append(gt_samples)
        #         # self.validation_step_ddim_samples.append(ddim_simples)

        # if self.current_epoch != 0 and (self.current_epoch) % 20 == 0: # full validation every 20 epochs

        #     print('\n###### Run full validation! ######\n')
        #     gt_samples, ddim_simples = self.generate_per_batch(batch, ddim_steps=self.ddim_steps, HW=None, state=self.state)
        #     self.validation_step_gt_samples.append(gt_samples)
        #     self.validation_step_ddim_samples.append(ddim_simples)


        # if batch_idx != 0: # validation is only performed on the first batch of each epoch (bs samples?)
        #     return
        
        # if self.validation_count % 5 == 0 and self.trainer.current_epoch != 0: # save image and best checkpoint
        #     self.full_validation(batch)
    
        # else: # not save image and best checkpoint
            # line 376, num_samples: repeatedly predict times for each prompt, limit: the number of prompts to sample in each batch
            # print('###### run partial validation! ######\n')
            # grid, all_samples, state = self.generate(batch, ddim_steps=self.ddim_steps, num_samples=3, limit=None) 
            # metric, metric_list = self.get_eval_metric(all_samples, avg=self.eval_avg) # avg=True
            # grid_imgs = Image.fromarray(grid.astype(np.uint8)) # big image
            # self.logger.log_image(key=f'samples_val_partial', images=[grid_imgs]) # save image
            # metric_dict = {f'val/{k}_partial':v for k, v in zip(metric_list, metric)} # key, value
            # print(metric_dict)
            # # self.logger.log_metrics(metric_dict, step=self.current_epoch)
            # self.log_dict(metric_dict, prog_bar=False, # should adjust x-axis in wandb
            #         logger=True, on_step=False, on_epoch=True) ### logger=True

            # if metric[-1] > self.run_full_validation_threshold: #  ssim > 0.15
            # self.full_validation(batch, state=state) # save images and save best val
        #     self.full_validation(batch) # run for each epoch 
        # self.validation_count += 1

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     _, loss_dict_no_ema = self.shared_step(batch)
    #     with self.ema_scope():
    #         _, loss_dict_ema = self.shared_step(batch)
    #         loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
    #     self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    #     self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    # def on_validation_epoch_end(self):
    #     all_preds = torch.stack(self.validation_step_outputs)
    #     # do something with all preds
    #     ...
    #     self.validation_step_outputs.clear()  # free memory

        # calculate each batch with each genetrated batch (num_samples) and get the mean
    def get_eval_metric(self, samples, avg=True): # (bs, gt+num_samples, 3, 512, 512)
        # metric_list = ['mse', 'pcc', 'ssim', 'psm'] # fid should be in metric only?
        metric_list = ['fid'] # fid should be in metric only?
        res_list = []
        
        gt_images = [img[0] for img in samples] # the first image within each batch
        gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
        samples_to_run = np.arange(1, len(samples[0])) if avg else [1] # num_of_genereted samples for each prompt
        for m in metric_list:
            res_part = [] # save for each metric
            for s in samples_to_run: # array([1, 2, 3, 4, 5])
                pred_images = [img[s] for img in samples] # for each prompt in batch get the generated image [s]
                pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
                # res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
                res = get_similarity_metric(pred_images, gt_images, method='metrics-only', metric_name=m)
                res_part.append(np.mean(res))
            res_list.append(np.mean(res_part))    

            # from torchmetrics.image.fid import FrechetInceptionDistance
            # fid = FrechetInceptionDistance(feature=64)
            # fid.update(rearrange(torch.Tensor(gt_images).to(torch.uint8), 'n w h c -> n c w h'), real=True)
            # fid.update(rearrange(torch.Tensor(pred_images).to(torch.uint8), 'n w h c -> n c w h'), real=False)
            # print(fid.compute())

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

        return res_list, metric_list   # value and key for each metric, calculate based on 5 sampling results




    def on_train_batch_end(self, *args, **kwargs): # False
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                first_stage_config, # dc_ldm.models.autoencoder.AutoencoderKL
                cond_stage_config, # dc_ldm.modules.encoders.modules.FrozenCLIPEmbedder
                num_timesteps_cond=None, # 1
                cond_stage_key="image", # 'count'
                cond_stage_trainable=True,
                concat_mode=True,
                cond_stage_forward=None,
                conditioning_key=None, # 'crossattn'
                scale_factor=1.0, # 0.18215
                scale_by_std=False, # False
                *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1) # 1
        self.scale_by_std = scale_by_std # False
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None) # None
        ignore_keys = kwargs.pop("ignore_keys", [])

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs) # 'crossattn' DDPM(pl.LightningModule)
        
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config) # dc_ldm.models.autoencoder.AutoencoderKL, obtained model.first_stage_model
        self.instantiate_cond_stage(cond_stage_config) # dc_ldm.modules.encoders.modules.FrozenCLIPEmbedder, obtained model.cond_stage_model
      
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None: # skip
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.train_cond_stage_only = False
        self.clip_tune = True
        if self.clip_tune:
            # self.image_embedder = FrozenImageEmbedder() # freeze the params and set the device to cuda
            # https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
            self.image_embedder, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device='cpu')
            # self.image_embedder, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', device='cpu')
            
            
            self.image_embedder = self.image_embedder.eval()
            for param in self.image_embedder.parameters():
                param.requires_grad = False

        self.cls_tune = False

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None): 
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"Setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config) # dc_ldm.models.autoencoder.AutoencoderKL
        self.first_stage_model = model.eval()

    def freeze_diffusion_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_diffusion_model(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_cond_stage(self):
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def unfreeze_cond_stage(self):
        for param in self.cond_stage_model.parameters():
            param.requires_grad = True
   
    def freeze_first_stage(self):
        self.first_stage_model.trainable = False
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def freeze_image_embedder(self):
        self.image_embedder.trainable = False
        for param in self.image_embedder.parameters():
            param.requires_grad = False

    def freeze_cond_cls(self):
        for name, param in self.cond_stage_model.named_parameters():
            if 'pooler' in name or 'classifier' in name:
                param.requires_grad = False

    def unfreeze_first_stage(self):
        self.first_stage_model.trainable = True
        for param in self.first_stage_model.parameters():
            param.requires_grad = True

    def freeze_whole_model(self):
        self.first_stage_model.trainable = False
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_whole_model(self):
        self.first_stage_model.trainable = True # AutoencoderKL
        for param in self.parameters(): # model.diffusion_model + first_stage_model + cond_stage_model + image_embedder
            param.requires_grad = True
        
    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model")
                self.cond_stage_model = None
                # self.be_unconditional = True
            
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                # self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        # self.cond_stage_model.eval()
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode): # False
            c, re_latent = self.cond_stage_model.encode(c)
            # c = self.cond_stage_model.encode(c)
        else: # idm_for_rna, line 168 
            c, re_latent = self.cond_stage_model(c)  # torch.Size([bs, 77, 768]) and torch.Size([bs, 2048, 256]) 
            # c = self.cond_stage_model(c)
        # return c
        return c, re_latent

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k) # batch, 'image' # [bs, 3, 512, 512]? # line 348
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        encoder_posterior = self.encode_first_stage(x) # get autoencoder encoder
        # print('encoder_posterior.shape')
        # print(encoder_posterior.shape)
        z = self.get_first_stage_encoding(encoder_posterior).detach() # [bs, 4, 64, 64]
        # print('z.shape')
        # print(z.shape)
        # print(cond_key)
        # print(self.cond_stage_key)
        # print(cond_key)
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox','fmri', 'rna', 'count']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            # print('get input')
            # print(not self.cond_stage_trainable)
            # print(force_c_encode)
            if not self.cond_stage_trainable or force_c_encode : # skip
                # print('get learned condition')
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c, re_latent = self.get_learned_conditioning(xc)
                    # c = self.get_learned_conditioning(xc)
                else:
                    c, re_latent = self.get_learned_conditioning(xc.to(self.device))
                    # c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c, batch['label'], batch['image_clip']]
        if return_first_stage_outputs: # skip
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        self.first_stage_model.eval()  ### Set the model to evaluation mode
        return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        self.freeze_first_stage() # freeze autoencoder
        # print('share step\'s get input')
        x, c, label, image_raw = self.get_input(batch, self.first_stage_key) # line 877, batch, 'image'
        # print('get input shape')
        # print('x.shape')
        # print(x.shape)
        # print('c.shape')
        # print(c.shape)
        if self.return_cond: # false
            loss, cc = self(x, c, label, image_raw)
            return loss, cc
        else:    
            loss = self(x, c, label, image_raw) ###
            return loss

    def forward(self, x, c, label, image_raw, *args, **kwargs):

        # x [bs, 4, 64, 64]
        # c [bs, 2048]
        # label [bs]
        # image_raw [bs, 3, 224, 224]

        # print(self.num_timesteps)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long() 
        # print('t.shape')
        # print(t.shape)

        if self.model.conditioning_key is not None: # 'crossattn'
            assert c is not None
            imgs = c # [4, 2048]
            if self.cond_stage_trainable: # True
                # c = self.get_learned_conditioning(c)
                c, re_latent = self.get_learned_conditioning(c) # [4, 2048] => [4, 77, 768], [4, 2048, 256]
                # print('c.shape')
                # print(c.shape)

        prefix = 'train' if self.training else 'val'
        loss, loss_dict = self.p_losses(x, c, t, *args, **kwargs) # [bs, 77, 768]
        
        # pre_cls = self.cond_stage_model.get_cls(re_latent)
        # rencon = self.cond_stage_model.recon(re_latent)

        if self.clip_tune:
            # biomedclip inputs image
            # with torch.no_grad():
            self.image_embedder.eval()

            image_embeds = self.image_embedder.encode_image(image_raw, normalize=True)  ### can normalize here
            # loss_clip = self.cond_stage_model.get_clip_loss(re_latent, image_embeds) # torch.Size([4, 128, 1024]), torch.Size([4, 768])
            loss_clip = self.cond_stage_model.get_clip_loss(c, image_embeds) # torch.Size([4, 77, 768]), torch.Size([4, 768])

            # # cosine similarity as logits
            # image_embeds, _, logit_scale = self.image_embedder(image_raw, None)  ### can normalize here
            # loss_clip = self.cond_stage_model.get_clip_ct_loss(c, image_embeds, logit_scale) # torch.Size([4, 77, 768]), torch.Size([4, 768])
            # here only let them to be similar

        
        # loss_recon = self.recon_loss(imgs, rencon)
        # loss_cls = self.cls_loss(label, pre_cls)

            loss += loss_clip

        # loss += loss_cls # loss_recon +  #(self.original_elbo_weight * loss_vlb)
        # loss_dict.update({f'{prefix}/loss_recon': loss_recon})
        # loss_dict.update({f'{prefix}/loss_cls': loss_cls})
            loss_dict.update({f'{prefix}/loss_clip': loss_clip})

        if self.cls_tune:
            pre_cls = self.cond_stage_model.get_cls(re_latent)
            loss_cls = self.cls_loss(label, pre_cls)
            # image_embeds = self.image_embedder(image_raw)
            # loss_clip = self.cond_stage_model.get_clip_loss(re_latent, image_embeds)
        # loss_recon = self.recon_loss(imgs, rencon)
        # loss_cls = self.cls_loss(label, pre_cls)
            loss += loss_cls
        # loss += loss_cls # loss_recon +  #(self.original_elbo_weight * loss_vlb)
        # loss_dict.update({f'{prefix}/loss_recon': loss_recon})
        # loss_dict.update({f'{prefix}/loss_cls': loss_cls})
            loss_dict.update({f'{prefix}/loss_cls': loss_cls})
                # if self.return_cond:
                    # return self.p_losses(x, c, t, *args, **kwargs), c
        # return self.p_losses(x, c, t, *args, **kwargs)

        loss_dict.update({f'{prefix}/loss_all': loss})

        if self.return_cond:
            return loss, loss_dict, c
        return loss, loss_dict ###



    # def recon_loss(self, )
    def recon_loss(self, imgs, pred):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = self.patchify(imgs)

        loss = (pred - imgs) ** 2
        loss = loss.mean()
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss
    
    def cls_loss(self, label, pred):
        return torch.nn.CrossEntropyLoss()(pred, label)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = torch.clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = torch.clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)
        # print('x_recon')
        # if isinstance(x_recon, tuple):
        #     print('is tuple')
        #     # print(len(x_recon))
        #     # print(x_recon[0].shape)
        # else:
        #     print(x_recon.shape)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # print('p_losses')
        # print('noise.shape')
        # print(noise.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print('x_noisy[0].shape')
        # print(x_noisy[0].shape)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps": # do this step
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3]) # noise loss for each batch [bs]
        # loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device) # all zeros
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar: # skip
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        ### add weight here 
        # self.l_simple_weight = 1

        loss = self.l_simple_weight * loss.mean() # loss_simple.mean()
        # can add weight here for loss_simple

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3)) # loss_simple
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean() # each timestep has a weight
        # loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        loss += (self.original_elbo_weight * loss_vlb) # weight = 0
        loss_dict.update({f'{prefix}/loss_unet': loss.item()}) # final use loss_simple as loss

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self): # do this step
        lr = self.learning_rate # 5.3e-05

        if self.train_cond_stage_only: # True
            print(f"{self.__class__.__name__}: Only optimizing conditioner params!")
            cond_parms = [p for n, p in self.named_parameters() 
                    if 'attn2' in n or 'time_embed_condtion' in n or 'norm2' in n]
            # cond_parms = [p for n, p in self.named_parameters() 
                    # if 'time_embed_condtion' in n]
            # cond_parms = []
            params = list(self.cond_stage_model.parameters()) + cond_parms   # only optimize conditioner + (attn2 + norm2 + time_embed_condtion in all model)
        
            for p in params:
                p.requires_grad = True

        else:
            params = list(self.model.parameters()) # diffusion_model => self.model, all model => self
            if self.cond_stage_trainable: # do this step
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                
                # cond_params = []
                # for name, param in self.cond_stage_model.named_parameters():
                #     if 'pooler' in name or 'classifier' in name:
                #         cond_params.append(param)

                params = params + list(self.cond_stage_model.parameters()) # full diffusion + conditioner
                # TODO: remove cls and pooler
            if self.learn_logvar: # False
                print('Diffusion model optimizing logvar')
                params.append(self.logvar) # all zero

        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler: # false
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
            
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)

        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)

        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + [c_concat], dim=1)
            cc = torch.cat([c_crossattn], dim=1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
