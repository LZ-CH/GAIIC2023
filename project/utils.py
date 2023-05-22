
import torch
import torch.nn as nn
from pathlib import Path
import time
import json
import numpy as np
import random
import copy
from collections import OrderedDict
from torch.nn.functional import normalize
import torch.distributed as dist
import math

def is_main_process():
    return dist.get_rank() == 0

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_parameters(model, pars):
    ret = [{'params': getattr(model, x).parameters()} for x in pars]
    print(ret)
    return ret

def output_tensor(x, precision=3):
    print(np.round(x.detach().cpu().numpy(), precision))

def to_device(data, device):
  if isinstance(data, torch.Tensor):
    data = data.to(device)
  elif isinstance(data, np.ndarray):
    data = to_device(torch.from_numpy(data), device)
  elif isinstance(data, tuple):
    data = tuple(to_device(item,device) for item in data)
  elif isinstance(data, list):
    data = list(to_device(item,device) for item in data)
  elif isinstance(data, dict):
    data = dict((k,to_device(v,device)) for k,v in data.items())
  else:
    raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.', type(data), data)
  return data

class Smoother():
    def __init__(self, window):
        self.window = window
        self.num = {}
        self.sum = {}
    def update(self, **kwargs):
        """
        为了调用方便一致,支持kwargs中有值为None的,会被忽略
        kwargs中一些值甚至可以为dict,也就是再套一层。
        示例: update(a=1, b=2, c={'c':1, 'd':3}),相当于update(a=1, b=2, c=1, d=3)
        如果值为参数的None的话忽略
        """
        values = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    values[x] = kwargs[key][x] #有可能会覆盖,如update(a=1,b={'a':2})
            else:
                values[key] = kwargs[key]
        for key in values:
            if values[key] is None:
                continue
            if key not in self.num:
                self.num[key] = []
                self.sum[key] = 0
            self.num[key].append(values[key])
            self.sum[key] += values[key]

            if len(self.num[key])>self.window:
                self.sum[key] -= self.num[key][-self.window-1]
            if len(self.num[key])>self.window*2:
                self.clear(key)
        pass
    def clear(self, key):
        del self.num[key][:-self.window]
    def value(self, key = None, mean=True):
        if mean:
            if key is None:
                return {key: self.sum[key] / min(len(self.num[key]),self.window) for key in self.num}
            return self.sum[key] / min(len(self.num[key]),self.window)
        if key is None:
            return {key: np.array(self.num[key]) for key in self.num}
        return np.array(self.sum[key])
    def keys(self):
        return self.num.keys()

class Step():
    def __init__(self):
        self.step = 0
        self.round = {}
    def clear(self):
        self.step = 0
        self.round = {}
    def forward(self, x):
        self.step += x
    def reach_cycle(self, mod, ignore_zero = True):
        now = self.step // mod
        if now==0 and ignore_zero:
            return False
        if mod not in self.round or self.round[mod]!=now: #新过了一个或多个cycle
            self.round[mod] = now
            return True
        return False
    def state_dict(self):
        return {'step': self.step, 'round':self.round}
    def load_state_dict(self, state):
        self.step = state['step']
        self.round = state['round']
    @property
    def value(self):
        return self.step

class Logger():
    def __init__(self, file_name, mode = 'w', buffer = 100):
        (Path(file_name).parent).mkdir(exist_ok = True, parents = True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer
    def log(self, *args, end='\n'):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y)+':'+str(x[y])+' ')
            else:
                self.fp.write(str(x)+' ')
        self.fp.write(end)
        self.cnt += 1
        if self.cnt>=self.buffer or time.time()-self.stamp>5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, 'a')
        pass
    def close(self):
        self.fp.close()

class Checkpoint():
    def __init__(self, **contents):
        """
        contents每个元素都需要有load_state_dict()方法
        """
        self.contents = contents
        self.contents['best_metrics'] = {}
    def update(self, file_path, logger = None, **kwargs):
        """
        根据metrics选择性地更新保存当前最好模型
        metrics: {metric_name: float 或 None},越大越好。None的话忽略
        file_path: 保存文件名,*.pt
        """
        metrics = {}
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                for x in kwargs[key]:
                    metrics[x] = kwargs[key][x] #有可能会覆盖,如update(a=1,b={'a':2})
            else:
                metrics[key] = kwargs[key]
        for metric in metrics:
            if metrics[metric] is None:
                continue
            if metric not in self.contents['best_metrics'] or metrics[metric]>self.contents['best_metrics'][metric]:
                self.contents['best_metrics'][metric] = metrics[metric]
                torch.save(self._get_contents(), file_path[:-3]+'_%s.pt'%metric)
                #torch.save(self.contents['optimizer'].state_dict(), file_path[:-3]+'_%s.pt'%metric)
                print('new best metric', metric, metrics[metric])
                if logger is not None:
                    logger.log('new best metric', metric, metrics[metric])
        pass
    def _get_contents(self):
        ret = {}
        for key in self.contents:
            if isinstance(self.contents[key], nn.DataParallel):
                ret[key] = self.contents[key].module.state_dict()
            elif hasattr(self.contents[key], 'state_dict'):
                ret[key] = self.contents[key].state_dict()
            else:
                ret[key] = self.contents[key]
        return ret
    def save(self, file_path):
        torch.save(self._get_contents(), file_path)
        
    def resume(self, file_path):
        memory = torch.load(file_path)
        self.contents['best_metrics'] = memory.pop('best_metrics')
        for key in memory:
            try:
                if key not in self.contents:
                    print('loaded key not in contents:', key)
                    continue
                if isinstance(self.contents[key], nn.DataParallel):
                    self.contents[key].module.load_state_dict(memory[key])
                else:
                    self.contents[key].load_state_dict(memory[key])
            except:
                print('warnning: can not load',key,'sucessfully')
        print('load checkpoint from ',file_path,'best metrics ',self.contents['best_metrics'])
        self.contents['best_metrics'] = {}
        pass
