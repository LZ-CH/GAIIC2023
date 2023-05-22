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
from copy import deepcopy
import math
random.seed(42)
def compare_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        return False
    for key in dict1:
        if key not in dict2 or torch.all(torch.eq(dict1[key], dict2[key])):
            return False
    return True
model_list_path = 'model_list.txt'

state_list = []
metric_list = []
with open(model_list_path, "r") as f:
    for line in f:
        temp = line.replace('\n','')
        if temp=='':
            continue
        state = torch.load(temp)
        
        is_same =False
        for s in state_list:
            if compare_dicts(state['model'],s['model']):
                is_same = True
        if is_same:
            continue
        
        state_list.append(state)
        metric_list.append(state['best_metrics'])


num = len(state_list)
for pkey in state_list[0]['model']:
    temp = 0.0
    for k in range(len(state_list)):
        temp = temp + state_list[k]['model'][pkey]/num
    state_list[0]['model'][pkey] = temp
print(metric_list)
state_list[0]['best_metrics']['metric_list'] = metric_list
torch.save(state_list[0],'model_list_swa.pt')
    
        
