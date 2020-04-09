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
#             print(k, len(v))
#         print(len(typ_list.keys()), v_num)
#         print(condition_token)                            
        
        model_class, tokenizer_class, pretrained_weights = (GPT2Model, GPT2Tokenizer, 'gpt2-xl')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        special_tokens = {'bos_token': '<START>', 'additional_special_tokens': condition_token}
        self.tokenizer.add_special_tokens(special_tokens)
        self.bert_model = model_class.from_pretrained(pretrained_weights)
        self.bert_model.resize_token_embeddings(len(self.tokenizer))

        
        self.emb_dim = 1600
        self.vocab_num = len(self.tokenizer)
        
        self.matrix = nn.Linear(self.emb_dim, self.vocab_num)
        
#         self.model_params = list(self.matrix.parameters())
        

    """Modeling"""
    def model_feeding(self, input_ids):
        output_vector = self.bert_model(input_ids)[0]
        voacb_logit = self.matrix(output_vector)
        
        return voacb_logit
        
        
    def cls_loss(self, model_out, label_ids):
        """
        model_out: (input_idxs, vocab_num) (logits)
        label_ids: (label_idxs)        
        """      
        pred_out = model_out[-len(label_ids):]
        loss = F.cross_entropy(pred_out, label_ids)
        
        return loss

