from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

# 超参设置
beam_size=10
source_length=256
target_length=128
epochs=10 
pretrained_model="microsoft/codebert-base" #Roberta: roberta-base
model_path = "./../model/java_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_batch_size=8
max_source_length=64
max_target_length=32
batch_size=64

config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
config = config_class.from_pretrained(pretrained_model)
tokenizer = tokenizer_class.from_pretrained(pretrained_model)

# 模型构建
encoder = model_class.from_pretrained(pretrained_model,config=config)    
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                beam_size=beam_size,max_length=32,
                sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

model.load_state_dict(torch.load(model_path))
model.to(device)


def generate_query(code_snippet: str) -> str:
    """[根据输入code_snippet生成输出query string]

    Args:
        code_snippet (str): [由空格分开的java代码字符串]
        "@Override public final String escape ( String s ) { checkNotNull ( s ) ; // GWT specific check (do not optimize) for ( int i = 0 ; i < s . length ( ) ; i ++ ) { char c = s . charAt ( i ) ; if ( ( c < replacementsLength && replacements [ c ] != null ) || c > safeMaxChar || c < safeMinChar ) { return escapeSlow ( s , i ) ; } } return s ; }"

    Returns:
        query(str): [生成的query字符串]
        "Escape the given string in + str +"
    """    
    if len(code_snippet) == 0:
        return ''
    
    source_tokens = tokenizer.tokenize(code_snippet)[:max_source_length-2]
    source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens)) 
    padding_length = max_source_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask+=[0]*padding_length

    target_tokens = tokenizer.tokenize("None")
    target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] *len(target_ids)
    padding_length = max_target_length - len(target_ids)
    target_ids+=[tokenizer.pad_token_id]*padding_length
    target_mask+=[0]*padding_length   

    all_source_ids = torch.tensor([source_ids], dtype=torch.long)
    all_source_mask = torch.tensor([source_mask], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids,all_source_mask)   

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    # 模型生成query
    p=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            preds = model(source_ids=source_ids,source_mask=source_mask)  
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)
    
    return p[0]


