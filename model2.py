import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import pandas as pd

from transformers import *

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        dataset = pd.read_csv('dataset/trainset.csv')
        columns = dataset.columns
        conditions = dataset[columns[0]]

        typ_list = {}
        for k in range(len(conditions)):
            cond_set = conditions[k].split(',')
            for m in range(len(cond_set)):
                cond_set[m] = cond_set[m].strip()
                pos = cond_set[m].index('[')
                if cond_set[m][:pos] in typ_list.keys():
                    typ_list[cond_set[m][:pos]].add(cond_set[m][pos+1:-1])
                else:            
                    typ_list[cond_set[m][:pos]] = {cond_set[m][pos+1:-1]}        

        condition_token = []
        v_num = 0
        for k, v in typ_list.items():
            v_num += len(v)
            condition_token.append('<'+k+'>')                        
        
        # model_class, tokenizer_class, pretrained_weights = (GPT2Model, GPT2Tokenizer, 'gpt2')
#         model_class, tokenizer_class, pretrained_weights = (GPT2Model, GPT2Tokenizer, '/data/private/GPT/openai-gpt2/base/')
        model_class, tokenizer_class, pretrained_weights = (GPT2LMHeadModel, GPT2Tokenizer, '/data/private/GPT/openai-gpt2/base/')        
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        special_tokens = {'bos_token': '<START>', 'additional_special_tokens': condition_token}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # pretrained model
        self.bert_model = model_class.from_pretrained(pretrained_weights)      
        
        # random model
#         configuration = GPT2Config()
#         self.bert_model = GPT2Model(configuration)
        
        
        self.bert_model.resize_token_embeddings(len(self.tokenizer))       

    """Modeling"""
    def model_feeding(self, input_ids):
        voacb_logit = self.bert_model(input_ids)[0] # (batch, length, voacb_size)
        
        return voacb_logit
        
        
    def cls_loss(self, model_out, label_ids):
        """
        model_out: (input_idxs, vocab_num) (logits)
        label_ids: (label_idxs)        
        """      
        pred_out = model_out[-len(label_ids):]
        loss = F.cross_entropy(pred_out, label_ids)
        
        return loss
        