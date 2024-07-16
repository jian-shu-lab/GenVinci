import datetime
import os

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

import pickle
import random
import subprocess
import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments
from geneformer import GeneformerPretrainer


seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# set local time/directories
# timezone = pytz.timezone("US/Eastern")


# dataset = 'he_st'

import sys
dataset = sys.argv[1]

datadir = './dataset/' + dataset
rootdir = "./output/" + dataset + "/mlm" ###

# set model parameters
# model type
model_type = "bert"
# max input size
max_input_size = 2**11  # 2048
# number of layers
num_layers = 6
# number of attention heads
num_attn_heads = 4
# number of embedding dimensions
num_embed_dim = 256
# intermediate size
intermed_size = num_embed_dim * 2
# activation function
activ_fn = "relu"
# initializer range, layer norm, dropout
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.02
hidden_dropout_prob = 0.02


# set training parameters
# total number of examples in Genecorpus-30M after QC filtering:
# num_examples = 29602  ### 27_406_208
# number gpus
num_gpus = 1  ### 12
# batch size for training and eval
geneformer_batch_size = 16 ### 16 ### 12
# max learning rate
max_lr = 5e-5   # 1e-3
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 2000 ### 10_000
# number of epochs
epochs = 20 ### 3
# optimizer
optimizer = "adamw"
# weight_decay
weight_decay = 0.001


# output directories
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"{datestamp}"

training_output_dir = f"{rootdir}/{run_name}/"
# logging_dir = f"{rootdir}/{run_name}/logs/"
# model_output_dir = f"{rootdir}/{run_name}/"


# ensure not overwriting previously saved model
model_output_file = os.path.join(training_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")


# make training and model output directories
# subprocess.call(f"mkdir {training_output_dir}", shell=True) 
# subprocess.call(f"mkdir {model_output_dir}", shell=True)   
if not os.path.exists(training_output_dir):
    os.makedirs(training_output_dir)
# if not os.path.exists(logging_dir):
#     os.makedirs(logging_dir)



# load gene_ensembl_id:token dictionary (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/token_dictionary.pkl)
with open("./src/pretrain/tokenizer/token_dictionary.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

# model configuration
config = {
    "hidden_size": num_embed_dim, # 256
    "num_hidden_layers": num_layers, # 6
    "initializer_range": initializer_range, # 0.02
    "layer_norm_eps": layer_norm_eps, # 1e-12
    "attention_probs_dropout_prob": attention_probs_dropout_prob, # 0.02
    "hidden_dropout_prob": hidden_dropout_prob, # 0.02
    "intermediate_size": intermed_size, # 512
    "hidden_act": activ_fn, # 'relu'
    "max_position_embeddings": max_input_size, # 2048
    "model_type": model_type, # 'bert'
    "num_attention_heads": num_attn_heads, # 4
    "pad_token_id": token_dictionary.get("<pad>"), # 0
    "vocab_size": len(token_dictionary),  # 25426, genes+2 for <mask> and <pad> tokens
}

config = BertConfig(**config)
model = BertForMaskedLM(config)

# TODO: here can load the pretrained model on Genecorpus-30M
model.load_state_dict(torch.load('./src/pretrain/geneformer/pytorch_model.bin'), strict=False) #
model = model.train()

# define the training arguments
training_args = {
    "learning_rate": max_lr, # 0.001
    "do_train": True,
    "do_eval": True, ### False
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn, # 'linear'
    "warmup_steps": warmup_steps, # 10000
    "weight_decay": weight_decay, # 0.001
    "per_device_train_batch_size": geneformer_batch_size, # 2
    "num_train_epochs": epochs, # 3
    "save_strategy": "epoch",

    "evaluation_strategy": "epoch", ### for val

    # "save_steps": np.floor(num_examples / geneformer_batch_size / 8),  # 8 saves per epoch
    "logging_steps": 1,
    "output_dir": training_output_dir, # 
    # "logging_dir": logging_dir, # 
    
    "load_best_model_at_end": True, # if used val
}
training_args = TrainingArguments(**training_args)

print("Continue training.")


# define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
    train_dataset=load_from_disk(datadir + '/train.dataset'), ###

    eval_dataset=load_from_disk(datadir + '/val.dataset'), ###
    
    # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/genecorpus_30M_2048_lengths.pkl)
    example_lengths_file=datadir + '/train_2048_lengths.pkl', ###
    token_dictionary=token_dictionary,
)

# train
trainer.train()

# save model
trainer.save_model(training_output_dir)

# deepspeed ./src/pretrain/pretrain_geneformer_w_deepspeed.py --deepspeed he_st

