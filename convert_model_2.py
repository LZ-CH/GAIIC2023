import torch
import os
import argparse

if __name__=='__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--input_file',type='str',default=None)
    arg.add_argument('--output_file',type='str',default=None)
    arg = arg.parse_args()
    #将bart转为嵌套使用的权重
    checkpoint = torch.load(arg.input_file)
    new_checkpoint = {}
    new_checkpoint['best_metrics'] = {}
    new_checkpoint['model'] = {}
    for k in checkpoint['model_state_dict']:
        new_checkpoint['model']['model.'+ k] = checkpoint['model_state_dict'][k]
        
    save = new_checkpoint
    torch.save(save,arg.output_file)

