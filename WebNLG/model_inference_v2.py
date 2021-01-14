from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import json
import xmltodict
import glob
from tqdm import tqdm
        
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

                # version 2.0
                if not isinstance(dict_data, list):
                    dict_data = [dict_data]
                for i in range(len(dict_data)):
                    y=dict_data[i]

                    self.category_list.append(y['@category'])

                    if 'train' in xml_file.split('/'):
                        self.modifiedtripleset_list.append(y['modifiedtripleset']['mtriple'])
                    else:
                        self.modifiedtripleset_list.append(y['modifiedtripleset']['otriple'])

                    z = y['lex']
                    if isinstance(z, list):
                        z = z[0]
                    self.text_list.append(z['text'])
        
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
    
from model import *

def main():    
    data_path = '/data/private/dataset/webnlg/data/v2.0/en/test/'
    webNLG_data = webNLG_DATASET(data_path)
    dataloader = DataLoader(webNLG_data, batch_size=1, shuffle=False, num_workers=4)
    
    my_model = webmodel().cuda()
    model_folder = '/data/private/WebNLG-models/simple_model/ver2/pretrained/try_3/'
    
    
    for i in range(5, 7):
        model_path = model_folder + str(i)
        my_model.load_state_dict(torch.load(model_path + '/model.bin'))
        my_model.eval()
        print(i)

#         f_w = open('./prediction/reference_'+str(i)+'.txt', 'w')
        f_p = open('./prediction/v2/try_3/prediction_'+str(i)+'.txt', 'w')
        
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