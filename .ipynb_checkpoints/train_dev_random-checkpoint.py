# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch

from model_random import *
my_model = mymodel().cuda()
my_model.train()
print('ok')

import pandas as pd
from torch.utils.data import Dataset, DataLoader
class e2eDataset(Dataset):
    def __init__(self, csv_file1, csv_file2, tokenizer):
        """
        Args:
            csv_file (string): csv 파일의 경로
        """
        self.dataset1 = pd.read_csv(csv_file1)
        self.dataset2 = pd.read_csv(csv_file2)
        
        self.columns1 = self.dataset1.columns
        self.columns2 = self.dataset2.columns
        
        self.conditions = list(self.dataset1[self.columns1[0]]) + list(self.dataset2[self.columns2[0]])
        self.sentences = list(self.dataset1[self.columns1[1]]) + list(self.dataset2[self.columns2[1]])
        self.tokenizer = tokenizer
        
        self.typ_list = {}
        for k in range(len(self.conditions)):
            cond_set = self.conditions[k].split(',')
            for m in range(len(cond_set)):
                cond_set[m] = cond_set[m].strip()
                pos = cond_set[m].index('[')
                if cond_set[m][:pos] in self.typ_list.keys():
                    self.typ_list[cond_set[m][:pos]].add(cond_set[m][pos+1:-1])
                else:            
                    self.typ_list[cond_set[m][:pos]] = {cond_set[m][pos+1:-1]}        

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        cond = self.conditions[idx]
        cond_set = cond.split(',')
        condition_string = ''
        for m in range(len(cond_set)):
            cond_set[m] = cond_set[m].strip()
            pos = cond_set[m].index('[')
            
            condition_string += '<' + cond_set[m][:pos] + '>' + cond_set[m][pos+1:-1] + ' '
        
        sen = self.sentences[idx]
        input_string = condition_string + '<START>' + sen
        input_ids = torch.tensor(self.tokenizer.encode(input_string, add_special_tokens=True))
        
        label_string = sen + '<|endoftext|>'
        label_ids = torch.tensor(self.tokenizer.encode(label_string, add_special_tokens=True))

        return input_ids, label_ids

## finetune bert_model
def main():
    e2e_dataset = e2eDataset(csv_file1='dataset/trainset.csv', csv_file2='dataset/devset.csv', tokenizer=my_model.tokenizer)
    dataloader = DataLoader(e2e_dataset, batch_size=1, shuffle=True, num_workers=4)    

    # Parameters:
    epoch = 8
    lr = 2e-5 # 2e-5
    max_grad_norm = 10
    num_training_steps = len(e2e_dataset)*epoch
    num_warmup_steps = len(e2e_dataset)

    ### In Transformers, optimizer and schedules are splitted and instantiated like this:
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr, eps=1e-06, weight_decay=0.01)
#     optimizer = AdamW(my_model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    for epoch in tqdm(range(epoch)):
        for i_batch, sample_batched in enumerate(dataloader):
            input_ids = sample_batched[0].squeeze(0).cuda()
            label_ids = sample_batched[1].squeeze(0).cuda()

            model_out = my_model.model_feeding(input_ids) # (batch, seq_len, emb_dim)    

            loss = my_model.cls_loss(model_out, label_ids)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()    
        
        """savining point"""
#         if (epoch+1)%2 == 0:
        save_model(epoch+1)
#     save_model('final') # final_model     
#     save_path = 'gen_model/base_devtrain_3/final/'
#     my_model.bert_model.save_pretrained(save_path)
#     my_model.tokenizer.save_pretrained(save_path)


def save_model(iteration):
    save_path = 'gen_model/random_init/try_1/'+str(iteration)+'/'
#     save_path = 'gen_model/random_init/'+str(iteration)+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(my_model.state_dict(), save_path+'model')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()        
