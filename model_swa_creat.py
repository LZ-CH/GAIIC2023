import torch
import random
import argparse
random.seed(42)
def compare_dicts(dict1, dict2):
    if len(dict1) != len(dict2):
        return False
    for key in dict1:
        if key not in dict2 or torch.all(torch.eq(dict1[key], dict2[key])):
            return False
    return True
if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model_path_list',nargs='+',type=str,required=True)
    args.add_argument('--model_output_path',type=str,required=True)
    args = args.parse_args()
    state_list = []
    metric_list = []
    model_path_list = args.model_path_list
    model_output_path = args.model_output_path
    for temp in model_path_list:
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
        metric_list.append(state['best_metrics']['cider'])           

    num = len(state_list)
    for pkey in state_list[0]['model']:
        temp = 0.0
        for k in range(len(state_list)):
            temp = temp + state_list[k]['model'][pkey]/num
        state_list[0]['model'][pkey] = temp
    print(metric_list)
    state_list[0]['best_metrics']['metric_list'] = metric_list
    torch.save(state_list[0],model_output_path)
    
        
