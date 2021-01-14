from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import json
import xmltodict
import glob
from tqdm import tqdm
import os
        
class webNLG_DATASET(Dataset):
    def __init__(self, data_path):
        self.category_list = []
        self.modifiedtripleset_list = []
        self.text_list = []            
        
        xml_files = glob.glob(data_path+'*')
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
        
    def __len__(self):
        return len(self.category_list)

    def __getitem__(self, idx): 
        triple_total = []
        if isinstance(self.modifiedtripleset_list[idx], list):
            for triple_list in self.modifiedtripleset_list[idx]:
                triple_total += triple_list.split('|')
        else:
            triple_total += self.modifiedtripleset_list[idx].split('|')
            
        triple = [x.strip() for x in triple_total]
        
        return self.category_list[idx], triple, self.text_list[idx]
    
import glob
from model_mat import *

def main():    
    data_path = '/data/private/WebNLG-models/chimera-master/data/WebNLG/raw/test/'
    webNLG_data = webNLG_DATASET(data_path)
    dataloader = DataLoader(webNLG_data, batch_size=1, shuffle=False, num_workers=4)
    
    my_model = webmodel().cuda()
    model_folder = '/data/private/WebNLG-models/simple_model/challenge/matrix/try_2/*'
    model_folders = glob.glob(model_folder)
    
    save_path = './prediction/challenge/try_2_mat/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i in range(len(model_folders)):
        model_path = model_folders[i]
        epoch = model_path.split('/')[-1]
        if 'run' in model_path:
            continue
        my_model.load_state_dict(torch.load(model_path + '/model.bin'))
        my_model.eval()
        print(i)

#         f_w = open('./prediction/reference_'+str(i)+'.txt', 'w')

        f_p = open(save_path+'prediction_'+str(epoch)+'.txt', 'w')
        
        c = 0
        for i_batch, sample_batched in tqdm(enumerate(dataloader)):
            c+=1
            cate, triple, text = sample_batched

            reference = text[0]
#             f_w.write(reference+'\n')

            input_tensor = my_model.make_tensor(cate, triple, '').squeeze(0)
            response = my_model.generate(input_tensor)
        
            if c == len(dataloader):
                f_p.write(response)
            else:
                f_p.write(response+'\n')

#         f_w.close()
        f_p.close()
        torch.cuda.empty_cache()
        
if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()                