import os
import numpy as np


class Config_Generative_Model:
    def __init__(self):

        # project parameters
        self.dataset = None
       
        self.pretrain_mlm_path = None ### updated in parser, mlm-pretrained model if any
        self.img_size = None

        self.seed = 2023 ###

        self.test_ratio = 0.1 ### for validation set
        
        self.tokenizer_path = './src/tokenizer' 
        self.output_path = './output' ### update with timestamp

        # resume check util
        self.checkpoint_path = None ### for resume training, if any
        self.pretrain_ldm_path = './pretrains' ### for original SD v-1-5

        np.random.seed(self.seed)

        # finetune parameters
        self.batch_size = 4 # updated in parser
        self.lr = 5e-5
        self.num_epoch = 500 ### updated in parser
        self.precision = 32 ###
        self.accumulate_grad = 1 ###
        self.global_pool = False ###
        self.use_time_cond = True ###
        self.clip_tune = True ### 
        self.cls_tune = False ###
        # self.cond_scale = 1.0

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250  ### updated in parser, revise for faster sampling
        self.HW = None
        self.eval_avg = True ###

