import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import pandas as pd

from transformers import *

class webmodel(nn.Module):
    def __init__(self):
        super(webmodel, self).__init__()
        self.gpu = True
        
        # model_class, tokenizer_class, pretrained_weights = (GPT2Model, GPT2Tokenizer, 'gpt2')
        model_class, tokenizer_class, pretrained_weights = (GPT2LMHeadModel, GPT2Tokenizer, '/data/private/GPT/openai-gpt2/base/')        
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    
        condition_token = ['<c>', '<tr1>', '<tr2>', '<tr3>']
        special_tokens = {'bos_token': '<S>', 'additional_special_tokens': condition_token}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # pretrained model
        self.model = model_class.from_pretrained(pretrained_weights)      
        
        # random model
#         configuration = GPT2Config()
#         self.model = GPT2LMHeadModel(configuration)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.emb_dim = 768
        self.vocab_num = len(self.tokenizer)        
        self.matrix = nn.Linear(self.emb_dim, self.vocab_num)
        
        self.END_idx_list = self.tokenizer.all_special_ids
        self.loss = torch.nn.CrossEntropyLoss()
        
    """Make input"""
    def make_tensor(self, cate, tripleset, text):
        tr_list = ['<tr1>', '<tr2>', '<tr3>']

        input_str = '<c> '
        input_str += cate[0]
        input_str += ' '


        triple_total = []
        if isinstance(tripleset, list):
            for tripleset_temp in tripleset:
                triple_total = tripleset_temp[0].split('|')
                triple_list = [x.strip() for x in triple_total]        

                for k in range(len(triple_list)):
                    triple = triple_list[k]
                    triple = triple_list[k]
                    input_str += tr_list[k] + ' '
                    input_str += triple.replace('_', ' ') # _이 너무 많음
                    input_str += ' '        
        else:
            triple_total += tripleset[0].split('|')           
            triple_list = [x.strip() for x in triple_total]   

            for k in range(len(triple_list)):
                triple = triple_list[k]
                input_str += tr_list[k] + ' '
                input_str += triple.replace('_', ' ') # _이 너무 많음
                input_str += ' '

        input_str += '<S>'
        if text is not '':
            input_str += ' '
            input_str += text[0]

        input_str = input_str.strip()
        input_tensor = self.tokenizer.encode(input_str, return_tensors="pt")
        
        if self.gpu:
            input_tensor = input_tensor.cuda()
        return input_tensor # (batch, length)           

    """Modeling"""
    def logit_feeding(self, torch_input):
#         out_logit = self.model(torch_input)[0] # (batch, length, voacb_size)
        
        hidden_vector = self.model.transformer(torch_input)[0] # (batch, length, hidden_emb)
        out_logit = self.matrix(hidden_vector)
        
        return out_logit
        
        
    def LM_loss(self, out_logit, target_len, label_ids):
        """
        out_logit: (input_idxs, vocab_num) (logits)
        target_len: int
        label_ids: (label_idxs)        
        """        
        pred_logit = out_logit[:,-target_len-1:,:].squeeze(0) # (len, vocab_num)
        if self.gpu:
            label_ids = label_ids.cuda()        
        loss_val = self.loss(pred_logit, label_ids)
        
        return loss_val
        
    """Generation"""
    def generate(self, input_tensor, max_len = 92):
#         prob_list = []
        with torch.no_grad():
            for _ in range(max_len):
                if input_tensor.shape[0] > 1024:
                    break
                out_logit = self.logit_feeding(input_tensor)
                pred_token_logit = out_logit[-1:,:].squeeze(0)

                prob = F.softmax(pred_token_logit, dim=0).sort(descending=True)
                pred_token_prob = prob[0][0]
                pred_token_id = prob[1][0]

    #             print("########", pred_token_id, pred_token_prob)

                if pred_token_id.item() in self.END_idx_list: ## self.tokenizer.eos_token_id외에도 다른 special token을 방지하기 위함
                    break
    #             prob_list.append(pred_token_prob)
                input_tensor = torch.cat([input_tensor, pred_token_id.unsqueeze(0)])    
                
            sep_pos = input_tensor.tolist().index(self.tokenizer.bos_token_id)
            response_tensor = input_tensor[sep_pos+1:]  
            response = self.tokenizer.decode(response_tensor)
        return response        