
import json
import glob
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
import numpy as np
from PIL import Image
import random
import time
import csv
import traceback

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
                wait = min(wait*2, 1000)
class DAEdataset(BaseDataset):
    def __init__(self, data_file_list, input_l, output_l,tokenizer):
        self.mask_token = tokenizer.mask_token
        self.description_token = tokenizer.description_token
        self.diagnosis_token = tokenizer.diagnosis_token
        self.clinical_token = tokenizer.clinical_token
        # self.clinical_token = tokenizer.sep_token        
        self.samples = []
        self.samples_prefix = []
        prefix = [self.description_token,self.clinical_token,self.diagnosis_token]
        for data_file in data_file_list:
            with open(data_file, 'r') as fp:
                reader = csv.reader(fp)
                for row in reader:
                    for k in range(1,len(row)):
                        if len(row[k].split())>0:
                            self.samples.append(row[k])
                            self.samples_prefix.append(prefix[k-1])
        self.tokenizer = tokenizer
        self.input_l = input_l
        self.output_l = output_l
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        source = self.samples[idx]
        source_corrupted =self.data_corruption(source)
        source_noisy = self.token_replace(source,p=0.8)
        
        return source_corrupted,source,source_noisy       
    
    def text_infilling(self,my_list,span_ratio=0.15):
        #text infill
        #span_start 和 span_end 之间的text会被mask,会保留end
        temp_list = my_list.copy()
        arr_list = temp_list
        span_max_length = int(span_ratio*len(arr_list))
        span_total_length = 0
        arr_list_span = []
        span_end = 0
        while span_total_length < span_max_length and len(arr_list)>1:
            span_start = random.randint(0,(len(arr_list)-1)//2)
            span_length =  np.random.poisson(lam=3)
            span_length = min(span_length,span_max_length-span_total_length)
            span_end = min(span_start+span_length,len(arr_list)-1)
            if random.random()<0.1:
                arr_list_span = arr_list_span + arr_list[0:span_start]+[str(random.randint(9,1640))]
            else:
                arr_list_span = arr_list_span + arr_list[0:span_start]+[self.mask_token]
            arr_list = temp_list[span_end:]
            span_total_length += span_length

        arr_list_span += arr_list
        return arr_list_span
    
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
    def data_corruption(self,my_str):
        arr_list = self.str2list(my_str)
        arr_list = self.text_infilling(arr_list)
        arr_list = self.sentence_shuffle(arr_list)
        arr_str = self.list2str(arr_list)
        return arr_str
    
    def list2str(self,my_list):
        my_str = ' '.join(str(x) for x in my_list)
        return my_str
    def str2list(self,mystr):
        mylist = [x for x in mystr.split()]
        return mylist

class DAEdataset_DC(BaseDataset):
    def __init__(self, data_file_list, input_l, output_l,tokenizer):
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
        
        self.description_token = tokenizer.description_token
        self.diagnosis_token = tokenizer.diagnosis_token
        self.clinical_token = tokenizer.clinical_token
        self.mask_token = tokenizer.mask_token
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        description = self.samples[idx][1]        

        clinical = ''
        clinical_corrupted = ''
        clinical_noisy = ''
        
        # source = self.description_token+' ' + description
        # target = self.diagnosis_token+' ' + diagnosis
        source_corrupted = self.data_corruption(description)
        source = description
        source_noisy = self.token_replace(description,p=0.8) 

        
        #clinical
        if len(self.samples[idx])==4:
            clinical = self.samples[idx][3]
            if clinical != '':
                clinical_corrupted = self.data_corruption(clinical)
                clinical_noisy = self.token_replace(clinical,p=0.8)
        if clinical !='':
            source_corrupted = source_corrupted+' ' + self.clinical_token+' ' + clinical_corrupted
            source = source+' ' + self.clinical_token+' ' + clinical
            source_noisy = source_noisy+' ' + self.clinical_token+' ' + clinical_noisy
              
        return source_corrupted.strip(),source.strip(),source_noisy.strip()      
    
    def text_infilling(self,my_list,span_ratio=0.15):
        #text infill
        #span_start 和 span_end 之间的text会被mask,会保留end
        temp_list = my_list.copy()
        arr_list = temp_list
        span_max_length = int(span_ratio*len(arr_list))
        span_total_length = 0
        arr_list_span = []
        span_end = 0
        while span_total_length < span_max_length and len(arr_list)>1:
            span_start = random.randint(0,(len(arr_list)-1)//2)
            span_length =  np.random.poisson(lam=3)
            span_length = min(span_length,span_max_length-span_total_length)
            span_end = min(span_start+span_length,len(arr_list)-1)
            
            arr_list_span = arr_list_span + arr_list[0:span_start]+[str(random.randint(9,1640))]
            
            arr_list_span = arr_list_span + arr_list[0:span_start]+[self.mask_token]
            arr_list = temp_list[span_end:]
            span_total_length += span_length

        arr_list_span += arr_list
        return arr_list_span
    
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
    def data_corruption(self,my_str):
        arr_list = self.str2list(my_str)
        arr_list = self.text_infilling(arr_list)
        arr_list = self.sentence_shuffle(arr_list)
        arr_str = self.list2str(arr_list)
        return arr_str
    
    def list2str(self,my_list):
        my_str = ' '.join(str(x) for x in my_list)
        return my_str
    def str2list(self,mystr):
        mylist = [x for x in mystr.split()]
        return mylist    

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
        # self.clinical_token = tokenizer.sep_token
        self.test = test
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        if self.test:
            description = self.samples[idx][1]        
            clinical = self.samples[idx][2]
        
            if self.agumentation:
                description = self.data_agumentation_forencoder(description,p=self.agumentation_p)  
                clinical = self.data_agumentation_forencoder(clinical,p=self.agumentation_p)  
            source = description
            if clinical != '':
                source = source +' '+self.clinical_token+' '+ clinical
            target = ''
            return source.strip(), target
        else:
            description = self.samples[idx][1]        
            diagnosis = self.samples[idx][2]
            clinical = ''
            if self.agumentation:
                description = self.data_agumentation_forencoder(description,p=self.agumentation_p)
            source = description
            target = diagnosis   
            #clinical
            if len(self.samples[idx])==4:
                clinical = self.samples[idx][3]
                if self.agumentation and clinical!='':
                    clinical = self.data_agumentation_forencoder(clinical,p=self.agumentation_p)
            if clinical != '':
                source = source+' ' + self.clinical_token+' ' + clinical
            if self.agumentation:
                target_noisy = self.data_agumentation_fordecoder(target,p=0.5)
            else:
                target_noisy = target
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
    def data_agumentation_forencoder(self,my_list,p=1.0):
        arr_list = my_list
        # arr_list = self.sentence_shuffle(arr_list,p=0.2)
        arr_list = self.token_replace(arr_list,p=0.5)

        return arr_list
    def data_agumentation_fordecoder(self,my_list,p=1.0):
        arr_list = my_list
        arr_list = self.token_replace(arr_list,p=p)
        return arr_list
    def sentence_shuffle(self,my_list,p=1.0):
        if random.random()<p:
            arr_list = my_list
            arr_list = self.str2list(arr_list)
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
            arr_list_shuffle = self.list2str(arr_list_shuffle)
            return arr_list_shuffle
        else:
            return my_list
    def list2str(self,my_list):
        my_str = ' '.join(str(x) for x in my_list)
        return my_str
    def str2list(self,my_str):
        my_list = [x for x in my_str.split()]
        return my_list