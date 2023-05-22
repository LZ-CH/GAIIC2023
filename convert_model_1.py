import torch
import os
import argparse

if __name__=='__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--input_file',type='str',default=None)
    arg.add_argument('--output_file',type='str',default=None)
    arg = arg.parse_args()
    #将权重转为bart能直接导入的格式
    checkpoint = torch.load(arg.input_file)
    new_checkpoint = {}

    for k in checkpoint['model']:
        new_checkpoint[k.replace('model.','',1)] = checkpoint['model'][k]
    save = {}
    save['model_state_dict'] = new_checkpoint
    torch.save(save,arg.output_file)

