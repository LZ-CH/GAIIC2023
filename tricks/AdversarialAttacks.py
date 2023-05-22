import torch
import copy
import random
'''
self.adv_param: 'emb'/'weight'
'''

class AdversarialAttacks_Switch():
    '''
    可用于随机切换对抗的方式
    '''
    def __init__(self,args,bool_name_list=['PGD','FGM','AWP']):
        # 将最开始设置为True的变量名称存储在列表中
        self.initial_true_vars = [arg_name for arg_name in bool_name_list if getattr(args,arg_name)]
        self.args = args
    def random_select(self):
        # 从最开始设置为True的变量中随机选择一个变量进行设置
        if self.initial_true_vars:
            for arg_name in self.initial_true_vars:
                setattr(self.args, arg_name, False)
            var_to_set_true = random.choice(self.initial_true_vars)
            setattr(self.args, var_to_set_true, True)
class PGD():
    def __init__(self, model,adv_param='emb'):
        # adv_param这个参数要换成你模型中embedding的参数名
        self.model = model
        self.adv_param = adv_param
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, is_first_attack=False):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self):
        # self.adv_param这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    self.grad_backup[name] = None
                else:
                    self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FGM():
    def __init__(self, model, adv_param='emb'):
        self.adv_param=adv_param
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.25):
        # self.adv_param这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # self.adv_param这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class AWP:
    def __init__(self, model,adv_param='weight', adv_lr=0.1, adv_eps=0.0001):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}
    def attack(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

class FreeLB(object):
    def __init__(self, adv_K, adv_lr, adv_init_mag, adv_max_norm=0., adv_norm_type='l2', base_model='bert'):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag    # adv-training initialize with what magnitude, 即我们用多大的数值初始化delta
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
    def attack(self, model, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids']
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        if self.adv_init_mag > 0:   # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)  # 扰动初始化
        loss, logits = None, None
        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init  # 累积一次扰动delta
            inputs['input_ids'] = None
            outputs = model(**inputs)
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()  # 备份扰动的grad
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)  # p='inf',无穷范数，获取绝对值最大者
                denorm = torch.clamp(denorm, min=1e-8)  # 类似np.clip，将数值夹逼到(min, max)之间
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()  # 计算该步的delta，然后累加到原delta值上(梯度上升)
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):  
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
        return loss, logits


