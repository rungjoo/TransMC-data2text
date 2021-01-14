# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch

from model_mat import *
my_model = webmodel().cuda()
my_model.train()
print('ok')

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import json
import xmltodict
import glob
        
class webNLG_DATASET(Dataset):
    def __init__(self, data_path):
        self.category_list = []
        self.modifiedtripleset_list = []
        self.text_list = []            
        
        xml_folders = glob.glob(data_path+'*')
        xml_folders.sort()
        
        for xml_folder in xml_folders:
            xml_roots = xml_folder+'/*'
            xml_files = glob.glob(xml_roots)
            xml_files.sort()
            
            for xml_file in xml_files:        
                with open(xml_file,'r') as f:
                    xmlString = f.read()
                dict_data = xmltodict.parse(xmlString)['benchmark']['entries']['entry']
                if not isinstance(dict_data, list):
                    dict_data = [dict_data]                

                # challenge version
                for i in range(len(dict_data)):
                    y=dict_data[i]
                    self.category_list.append(y['@category'])
                    
                    self.modifiedtripleset_list.append(y['modifiedtripleset']['mtriple'])
                    z = y['lex']
                    if isinstance(z, list):
                        z = z[0]
                    self.text_list.append(z['#text'])

                
                # version 2.0
#                 for i in range(len(dict_data)):
#                     y=dict_data[i]

#                     self.category_list.append(y['@category'])

#                     if 'test' in xml_file.split('/'):
#                         self.modifiedtripleset_list.append(y['modifiedtripleset']['otriple'])
#                     else:
#                         self.modifiedtripleset_list.append(y['modifiedtripleset']['mtriple'])

#                     z = y['lex']
#                     if isinstance(z, list):
#                         z = z[0]
#                     self.text_list.append(z['text'])
        
    def __len__(self):
        return len(self.category_list)

    def __getitem__(self, idx): 
        
        return self.category_list[idx], self.modifiedtripleset_list[idx], self.text_list[idx]
    
## finetune GPT_model
def main():
    data_path = '/data/private/WebNLG-models/chimera-master/data/WebNLG/raw/train/'
    webNLG_data = webNLG_DATASET(data_path)
    dataloader = DataLoader(webNLG_data, batch_size=1, shuffle=False, num_workers=4)
    
#     data_path_dev = '/data/private/dataset/webnlg/data/v2.0/en/dev/'
#     webNLG_data_dev = webNLG_DATASET(data_path_dev)
#     dataloader_dev = DataLoader(webNLG_data_dev, batch_size=1, shuffle=True, num_workers=4)    
    
    # Tensorboard
    writer = SummaryWriter('./simple_model/challenge/matrix/try_2/runs')

    # Parameters:
    epoch = 16
    lr = 6.25e-5 # 2e-5
    max_grad_norm = 10
    num_training_steps = len(dataloader)*epoch
    num_warmup_steps = len(dataloader)

    ### In Transformers, optimizer and schedules are splitted and instantiated like this:
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    
    count_num = 0
    for epoch in tqdm(range(epoch)):
        for i_batch, sample_batched in enumerate(dataloader):
            count_num += 1
            cate, triple, text = sample_batched
            input_tensor = my_model.make_tensor(cate, triple, text)
            
            out_logit = my_model.logit_feeding(input_tensor)
            target_idx = my_model.tokenizer.encode(text[0])
            target_len = len(target_idx)
            
            label_idxs = torch.tensor(target_idx + [my_model.tokenizer.eos_token_id]) # (len)
            
            loss = my_model.LM_loss(out_logit, target_len, label_idxs)            
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()    

            writer.add_scalar('LM_loss', loss.item(), count_num)
            
#         for i_batch, sample_batched in enumerate(dataloader_dev):
#             count_num += 1
#             cate, triple, text = sample_batched
#             input_tensor = my_model.make_tensor(cate, triple, text)
            
#             out_logit = my_model.logit_feeding(input_tensor)
#             target_idx = my_model.tokenizer.encode(text[0])
#             target_len = len(target_idx)
            
#             label_idxs = torch.tensor(target_idx + [my_model.tokenizer.eos_token_id]) # (len)
            
#             loss = my_model.LM_loss(out_logit, target_len, label_idxs)            
                
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()    

#             writer.add_scalar('LM_loss', loss.item(), count_num)

        """savining point"""
        if (epoch+1) > 5 and (epoch+1)%2 == 0:
            save_model(epoch+1)

        
def save_model(iteration):
    save_path = './simple_model/challenge/matrix/try_2/'+str(iteration)+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(my_model.state_dict(), save_path+'model.bin')
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()        