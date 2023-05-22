from torch.utils.data import Dataset,DataLoader
from transformers import BartForConditionalGeneration, BertTokenizer,AutoConfig
import numpy as np
import time
import csv
import traceback
import torch
import os
import random
from comput_cider import comput_cider_forcsv
from BlockShuffle import ChunkedBatchSampler

from models import GenerationModel,GenerationModel_Pretrain
# from FusionGenerationModel import FusionGenerationModel as GenerationModel
from utils import to_device, Checkpoint, Step, Smoother, Logger
from tqdm import tqdm
random.seed(42)
print(os.getcwd())
root_path = os.getcwd()

TOKENIZER = 'bart-base-chinese-gaiic'
MODEL_NAME = 'bart-base-chinese-gaiic'
CHECKPOINT = None
CHECKPOINT = '../checkpoint/bart-base/bart-base-MLM-DAE-Switch/bart-base-switch-avg.pt'

class BaseDataset(Dataset):
    def _try_getitem(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        wait = 0.1
        while True:
            try:
                ret = self._try_getitem(idx)
                return ret
            except KeyboardInterrupt:
                break
            except (Exception, BaseException) as e:
                exstr = traceback.format_exc()
                print(exstr)
                print('read error, waiting:', wait)
                time.sleep(wait)
                wait = min(wait * 2, 1000)

class Sep2SepDataset(BaseDataset):
    def __init__(self, data_file_list, input_l, output_l, tokenizer,agumentation = False, test=False):
        if not isinstance(data_file_list,list):
            data_file_list =[data_file_list]   
        self.samples = []
        for data_file in data_file_list:
            with open(data_file, 'r') as fp:
                reader = csv.reader(fp)
                self.samples.extend([row for row in reader])
        self.input_l = input_l
        self.output_l = output_l
        self.tokenizer = tokenizer
        self.agumentation = agumentation
        self.agumentation_p = 0.2
        
        self.description_token = tokenizer.description_token
        self.diagnosis_token = tokenizer.diagnosis_token
        self.clinical_token = tokenizer.clinical_token
        self.test = test
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        if self.test:
            description = self.samples[idx][1] 
            clinical = self.samples[idx][-1]
            if self.agumentation:
                description = self.data_agumentation(description,p=self.agumentation_p)  
                clinical = self.data_agumentation(clinical,p=self.agumentation_p)  
            # source = self.description_token +' '+ description+' ' + self.clinical_token +' '+ clinical
            source = description
            if clinical != '':
                source = source +' '+self.clinical_token+' '+ clinical
            target = ''
            return source.strip(), target,idx
        else:
            description = self.samples[idx][1]        
            diagnosis = self.samples[idx][2]
            clinical = ''
            if self.agumentation:
                description = self.data_agumentation(description,p=self.agumentation_p)
            # source = self.description_token+' ' + description
            # target = self.diagnosis_token+' ' + diagnosis
            source = description
            target = diagnosis   
            #clinical
            if len(self.samples[idx])==4:
                clinical = self.samples[idx][3]
                if self.agumentation and clinical != '':
                    clinical = self.data_agumentation(clinical,p=self.agumentation_p)
            if clinical != '':
                source = source+' ' + self.clinical_token+' ' + clinical
            target_noisy = self.data_agumentation(target,p=self.agumentation_p)
            return source.strip(),target,target_noisy
    
    def token_replace(self,my_str,p=1.0,p_drop=0.15):
        #text infill
        if random.random()>p:
            return my_str
        arr_list = [int(s) for s in my_str.split()]
                
        #drop as BERT
        arr = np.array(arr_list)
        mask = np.random.rand(len(arr)) < p_drop
        
        random_words = np.random.randint(size=arr.shape, high=1640,low=9)
        arr = np.where(mask,random_words,arr)

        new_list = list(arr)
        new_str = ' '.join(str(x) for x in new_list)
        return new_str
    def data_agumentation(self,my_list,p=1.0):
        arr_list = my_list
        if random.random()<p:
            arr_list = self.token_replace(arr_list) 
            # arr_list = self.str2list(my_list)
            # arr_list = self.sentence_shuffle(arr_list)            
            # arr_list = self.list2str(arr_list)
        return arr_list
    def sentence_shuffle(self,my_list):
        arr_list = my_list.copy()
        sentence_list=[]
        sentence_end = ['10','11']
        item=[]
        for i in range(len(arr_list)):
            item.append(arr_list[i])
            if arr_list[i] in sentence_end or i==len(arr_list)-1:
                sentence_list.append(item.copy())
                item = []
        random.shuffle(sentence_list)
        arr_list_shuffle = []
        for k in sentence_list:
            arr_list_shuffle.extend(k)
        return arr_list_shuffle
    def list2str(self,my_list):
        my_str = ' '.join(str(x) for x in my_list)
        return my_str
    def str2list(self,my_str):
        my_list = [x for x in my_str.split()]
        return my_list

def invoke(input_data_path, output_data_path):
    tokenizer_name = TOKENIZER
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    test_data = Sep2SepDataset(input_data_path, 200, 80,tokenizer=tokenizer,test=True)
    # test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=1, drop_last=False)
    
    test_chunked_batch_sampler = ChunkedBatchSampler(test_data, 128, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_data,  num_workers=1, batch_sampler=test_chunked_batch_sampler)
    model = GenerationModel(tokenizer_name=tokenizer_name,model_name = MODEL_NAME,load_pretrained=False)
    if CHECKPOINT is not None:
        checkpoint = Checkpoint(model = model)
        checkpoint.resume(CHECKPOINT)
    
    model = model.cuda()    
    model.eval()
    try:
        for mo in model.model_list:
            mo.eval()
    except:
        pass
    res = {}
    for source,_,idx in tqdm(test_loader):
        source = list(source)
        pred = model(source)
        pred = pred.cpu().numpy()
        pred = tokenizer.batch_decode(pred,skip_special_tokens=True)

        for i in range(len(pred)):
            # res.append(pred[i])
            res[int(idx[i])] = pred[i]

    with open(output_data_path, 'w') as file:
        writer = csv.writer(file)
        for idx in range(len(res)):
            writer.writerow([idx, res[idx]])


if __name__ == '__main__':
    input_data_path = "../gaiic_dataset/semi_valid_split.csv"
    output_data_path = "./semi_valid_split_result.csv"
    
    invoke(input_data_path,output_data_path)
    comput_cider_forcsv(output_data_path,input_data_path)