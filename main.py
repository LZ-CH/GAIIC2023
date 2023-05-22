
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import csv

from utils import to_device, Checkpoint, Step, Smoother, Logger
from models import GenerationModel,GenerationModel_Pretrain
from dataset import Sep2SepDataset,DAEdataset,DAEdataset_DC
from config import build_config
from losses import CE,soft_label_loss,comput_R_drop_loss,simcse_unsup_loss,comput_KL_loss
from evaluate import CiderD
from utils import cosine_lr_schedule
import os
import random
from transformers.optimization import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from transformers import BertTokenizer
from tricks.EMA import ExponentialMovingAverage
from tricks.AdversarialAttacks import PGD,FGM,AWP,AdversarialAttacks_Switch
from tricks.Optimizer import Lion
from tricks.BlockShuffle import ChunkedBatchSampler
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.cuda.amp import autocast, GradScaler
import contextlib
from tricks.ChildTuningOptimizer import ChildTuningAdamW
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@torch.no_grad()
def evaluate(model, loader, beam=1, n=-1):
    model.eval()
    metrics = Smoother(100)
    res, gts = [], {}
    tot = 0
    fp = open(os.path.join(config.output_dir,'pred_val.csv'), 'w', newline='')
    writer = csv.writer(fp)
    for (source,targets,_) in tqdm(loader):
        source,targets = list(source),list(targets)
        if n>0 and tot>n:
            break        
        with torch.no_grad():
            pred = model(source)
        pred = tokenizer.batch_decode(pred.cpu().numpy(),skip_special_tokens=True)
        
        targets = [t.replace(tokenizer.diagnosis_token,'').strip() for t in targets]
        for i in range(len(pred)):
            res.append({'image_id':tot, 'caption': [pred[i]]})
            preds = [tot, pred[i]]
            gts[tot] = [targets[i]]
            writer.writerow(preds)
            tot += 1
    fp.close()
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts , res)
    metrics.update(cider = cider_score)
    print(metrics.value())
    return metrics

def pretrain(eval_epoch = 3):
    config.output_dir = config.output_dir+'-pretrain'
    # train_data = Sep2SepDatasetForpretrain(config.pretrain_file_list, config.input_l, config.output_l,tokenizer=tokenizer)
    # valid_data = Sep2SepDatasetForpretrain([config.valid_file], config.input_l, config.output_l,tokenizer=tokenizer)     
    train_data = DAEdataset_DC(config.pretrain_file_list, config.input_l, config.output_l,tokenizer=tokenizer)
    valid_data = DAEdataset_DC([config.valid_file], config.input_l, config.output_l,tokenizer=tokenizer) 
    
    # train_loader = DataLoader(train_data, batch_size=config.pretrain_batch, shuffle=True, num_workers=1, drop_last=True)
    # valid_loader = DataLoader(valid_data, batch_size=config.valid_batch, shuffle=False, num_workers=1, drop_last=False)
    
    train_chunked_batch_sampler = ChunkedBatchSampler(train_data, config.pretrain_batch, drop_last=False, shuffle=True)
    valid_chunked_batch_sampler = ChunkedBatchSampler(valid_data, config.valid_batch, drop_last=False, shuffle=False)
    train_loader = DataLoader(train_data, num_workers=1, batch_sampler=train_chunked_batch_sampler)
    valid_loader = DataLoader(valid_data,  num_workers=1, batch_sampler=valid_chunked_batch_sampler)
    
    step = Step()
    model = GenerationModel_Pretrain(config)
    checkpoint = Checkpoint(model = model, step = step)
    if config.pretrained_checkpoint is not None:
        checkpoint.resume(config.pretrained_checkpoint)
    model = model.cuda()

    # num_training_steps = num_warmup_steps + (num_training_steps-num_warmup_steps)/(1-config['min_lr']/config['lr'])
    
    num_training_steps = config.pretrain_epoch*len(train_loader)
    num_warmup_steps=int(config.warmup_ratio*num_training_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr)
    lr_schedule = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps,
                                                last_epoch=-1)
    start_epoch = 0
    start_epoch = config.pretrain_epoch-1
    
    train_loss = Smoother(100)
    logger = Logger(config.output_dir+'/log.txt', 'a')
    logger.log(config)
    writer = SummaryWriter(config.output_dir)
    
    scaler = GradScaler()
    amp_cm = autocast if config.amp else contextlib.nullcontext
    Path(config.output_dir).mkdir(exist_ok=True, parents=True)
    if config.ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = config.finetune_batch*config.accumulation_steps* config.ema_steps / config.finetune_epoch
        alpha = 1.0 - config.ema_decay
        alpha = min(1.0, alpha * adjust)
        logger.log('EMA decay:',1-alpha)
        print('EMA decay:',1-alpha)
        model_ema = ExponentialMovingAverage(model, device='cuda', decay=1.0 - alpha)
        step_ema = Step()
        checkpoint_ema = Checkpoint(model = model_ema.module,step = step_ema)

    for epoch in range(start_epoch, config.pretrain_epoch):
        model.train()
        print('epoch:', epoch,'lr:',optimizer.param_groups[0]['lr'])
        logger.log('new epoch', epoch)
        for iter, (source,target,target_noisy) in enumerate(tqdm(train_loader, dynamic_ncols=True)):

            source, target = list(source),list(target)
            target_noisy = list(target_noisy)
            step.forward(len(source))
            
            with amp_cm():
                pred,loss = model(source,target,target_noisy)
            if config.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            train_loss.update(loss={'celoss':loss.item()})
            
            if (iter+1) % config.accumulation_steps_pretrain == 0:
                if config.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step() #优化一次
                lr_schedule.step()
                optimizer.zero_grad() #清空梯度
                if config.ema and (iter+1)%(config.accumulation_steps*config.ema_steps)==0:
                    model_ema.update_parameters(model)
                    if epoch < int(config.ema_warmup_ratio*config.pretrain_epoch):
                        # Reset ema buffer to keep copying weights during warmup period
                        model_ema.n_averaged.fill_(0)
            if step.value%100==0:
                logger.log(step.value, train_loss.value())
                
        if (epoch+1)%eval_epoch==0 or epoch==config.pretrain_epoch-1:
            # checkpoint.save(config.output_dir+'/model_%d.pt'%epoch)
            metrics = evaluate(model, valid_loader)
            logger.log('valid', step.value, metrics.value())
            writer.add_scalars('valid metric', metrics.value(), step.value)
            checkpoint.update(config.output_dir+'/model.pt', metrics = metrics.value())
            if config.ema and epoch >= int(config.ema_warmup_ratio*config.pretrain_epoch):
                metrics_ema = evaluate(model_ema.module, valid_loader)
                logger.log('valid', step_ema.value, metrics_ema.value())
                writer.add_scalars('valid metric of ema', metrics_ema.value(), step_ema.value)
                checkpoint_ema.update(config.output_dir+'/model_ema.pt', metrics = metrics_ema.value())
        checkpoint.save(config.output_dir+'/model_last.pt')
        if config.ema and epoch >= int(config.ema_warmup_ratio*config.pretrain_epoch):
            checkpoint_ema.save(config.output_dir+'/model_ema_last.pt')
    logger.close()
    writer.close()

def only_eval():
    valid_data = Sep2SepDataset(config.valid_file, config.input_l, config.output_l,tokenizer=tokenizer,agumentation=False)
    # valid_loader = DataLoader(valid_data, batch_size=config.valid_batch, shuffle=False, num_workers=1, drop_last=False)
    
    valid_chunked_batch_sampler = ChunkedBatchSampler(valid_data, config.valid_batch, drop_last=False, shuffle=False)
    valid_loader = DataLoader(valid_data,  num_workers=1, batch_sampler=valid_chunked_batch_sampler)    
    step = Step()
    model = GenerationModel(config,load_pretrained=False)
    checkpoint = Checkpoint(model = model, step = step)
    if config.pretrained_checkpoint is not None:
        checkpoint.resume(config.pretrained_checkpoint)
    model = model.cuda()
    model.eval()
    os.makedirs(config.output_dir,exist_ok=True)
    evaluate(model, valid_loader)

def finetune(eval_epoch = 1):
    
    train_data = Sep2SepDataset(config.train_file, config.input_l, config.output_l,tokenizer=tokenizer,agumentation=True)
    valid_data = Sep2SepDataset(config.valid_file, config.input_l, config.output_l,tokenizer=tokenizer,agumentation=False)

    # train_loader = DataLoader(train_data, batch_size=config.finetune_batch, shuffle=True, num_workers=1, drop_last=True)
    # valid_loader = DataLoader(valid_data, batch_size=config.valid_batch, shuffle=False, num_workers=1, drop_last=False)
    train_chunked_batch_sampler = ChunkedBatchSampler(train_data, config.finetune_batch, drop_last=False, shuffle=True)
    valid_chunked_batch_sampler = ChunkedBatchSampler(valid_data, config.valid_batch, drop_last=False, shuffle=False)
    train_loader = DataLoader(train_data, num_workers=1, batch_sampler=train_chunked_batch_sampler)
    valid_loader = DataLoader(valid_data,  num_workers=1, batch_sampler=valid_chunked_batch_sampler)
    step = Step()
    model = GenerationModel(config,load_pretrained=False)
    checkpoint = Checkpoint(model = model, step = step)
    if config.pretrained_checkpoint is not None:
        checkpoint.resume(config.pretrained_checkpoint)
    model = model.cuda()    
    num_training_steps = config.finetune_epoch*len(train_loader)
    num_warmup_steps=int(config.warmup_ratio*num_training_steps)
    # num_training_steps = num_warmup_steps + (num_training_steps-num_warmup_steps)/(1-config['min_lr']/config['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr)
    # optimizer = ChildTuningAdamW(model.parameters(), lr=config.finetune_lr)
    lr_schedule = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                  num_warmup_steps=num_warmup_steps,
                                                  num_training_steps=num_training_steps,
                                                  last_epoch=-1)
    start_epoch = 0
    start_epoch = config.finetune_epoch-1
    
    
    train_loss = Smoother(100)
    logger = Logger(config.output_dir+'/log.txt', 'a')
    logger.log(config)
    writer = SummaryWriter(config.output_dir)
    
    Path(config.output_dir).mkdir(exist_ok=True, parents=True)
    if config.PGD:
        pgd = PGD(model,adv_param=config.adv_param)
    if config.FGM:
        fgm = FGM(model,adv_param=config.adv_param)
    if config.AWP:
        awp = AWP(model,adv_param=config.adv_param)

    model_swa = None
    if config.swa:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        # adjust = config.finetune_batch*config.accumulation_steps* config.swa_steps / config.finetune_epoch
        # alpha = 1.0 - config.swa_decay
        # alpha = min(1.0, alpha * adjust)
        # model_swa = ExponentialMovingAverage(model, device='cuda', decay=1.0 - alpha)
        model_swa = torch.optim.swa_utils.AveragedModel(model,device='cuda')
        step_swa = Step()
        checkpoint_swa = Checkpoint(model = model_swa.module,step = step_swa)
    if config.ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = config.finetune_batch*config.accumulation_steps* config.ema_steps / config.finetune_epoch
        alpha = 1.0 - config.ema_decay
        alpha = min(1.0, alpha * adjust)
        logger.log('EMA decay:',1-alpha)
        print('EMA decay:',1-alpha)
        model_ema = ExponentialMovingAverage(model, device='cuda', decay=1.0 - alpha)
        step_ema = Step()
        checkpoint_ema = Checkpoint(model = model_ema.module,step = step_ema)
        model_ema.eval()
    scaler = GradScaler()
    amp_cm = autocast if config.amp else contextlib.nullcontext
    if config.Switch:
        attack_switch = AdversarialAttacks_Switch(config,bool_name_list=['PGD','FGM','AWP'])
    for epoch in range(start_epoch, config.finetune_epoch):
        model.train()
        print('epoch:', epoch,'lr:',optimizer.param_groups[0]['lr'])
        logger.log('new epoch', epoch)
        for iter, (source,target,target_noisy) in enumerate(tqdm(train_loader, dynamic_ncols=True)):
            if config.Switch:
                attack_switch.random_select()
            source,target = list(source),list(target)
            target_noisy = list(target_noisy)
            step.forward(len(source))
            with amp_cm():
                if config.R_drop:
                    decoder_labels = tokenizer(target, max_length = config.input_l,truncation=True,add_special_tokens=True,padding='longest',return_tensors='pt')
                    decoder_input_ids = decoder_labels['input_ids'].cuda()
                    # decoder_input_ids = decoder_input_ids[:,1:]
                    source_rdrop = 2*source
                    target_rdrop = 2*target
                    target_noisy_rdrop = 2*target_noisy
                    encoder_last_hidden_state,pred,loss = model(source_rdrop,target_rdrop,target_noisy_rdrop,return_encoder_last_hidden_state=True)
                    
                    pq = torch.split(pred,pred.shape[0]//2,dim=0)
                    
                    # encoder_last_hidden_state_pq = torch.split(encoder_last_hidden_state,encoder_last_hidden_state.shape[0]//2,dim=0)
                    R_drop_loss =0.
                    p = pq[0]
                    q = pq[1]
                    R_drop_loss += comput_R_drop_loss(p,q,decoder_input_ids,ignore_index=tokenizer.pad_token_id)
                    loss += config.alpha*R_drop_loss 
                else:
                    pred,loss = model(source,target,target_noisy)
            
            if config.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if epoch > config.attack_start_epoch-1 and config.PGD:
                pgd.backup_grad() # 保存正常的grad
                # 对抗训练
                for t in range(config.attack_step):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    
                    if t != config.attack_step-1:
                        optimizer.zero_grad()
                    else:
                        pgd.restore_grad() # 恢复正常的grad
                    with amp_cm():
                        pred,loss_pgd = model(source,target,target_noisy)
                    if config.amp:
                        scaler.scale(loss_pgd).backward()
                    else:
                        loss_pgd.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore() # 恢复embedding参数
            if epoch > config.attack_start_epoch-1 and config.FGM:
                fgm.attack() # embedding被修改了
                # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                with amp_cm():
                    pred,loss_fgm = model(source,target,target_noisy)
                if config.amp:
                    scaler.scale(loss_fgm).backward()
                else:
                    loss_fgm.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                fgm.restore() # 恢复Embedding的参数
            if epoch > config.attack_start_epoch-1 and config.AWP:
                awp.backup_grad()# 保存正常的grad
                awp.attack() # embedding被修改了
                # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
                with amp_cm():
                    pred,loss_awp= model(source,target,target_noisy)
                if config.amp:
                    loss_awp =scaler.scale(loss_awp).backward()
                else:
                    loss_awp.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
                awp.restore() # 恢复Embedding的参数
            train_loss.update(loss={'celoss':loss.item()})

            if (iter+1) % config.accumulation_steps == 0:
                
                #grad clip
                if config.amp:
                    scaler.unscale_(optimizer)
                if hasattr(optimizer, "clip_grad_norm"):
                    # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                    optimizer.clip_grad_norm(config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

                if config.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step() #优化一次
                lr_schedule.step()
                optimizer.zero_grad() #清空梯度
                if config.ema and (iter+1)%(config.accumulation_steps*config.ema_steps)==0:
                    model_ema.update_parameters(model)
                    if epoch < int(config.ema_warmup_ratio*config.finetune_epoch):
                        # Reset ema buffer to keep copying weights during warmup period
                        model_ema.n_averaged.fill_(0)
            if step.value%100==0:
                logger.log(step.value, train_loss.value())
        print(step.value,train_loss.value())
        if config.swa:
            model_swa.update_parameters(model_ema.module)
            if epoch < int(config.swa_warmup_ratio*config.finetune_epoch):
                # Reset ema buffer to keep copying weights during warmup period
                model_swa.n_averaged.fill_(0)
        
        # 清空未被使用的显存缓存
        torch.cuda.empty_cache()
        if epoch%eval_epoch==0 or epoch==config.finetune_epoch-1:
            # checkpoint.save(config.output_dir+'/model_%d.pt'%epoch)
            metrics = evaluate(model, valid_loader)
            logger.log('valid', step.value, metrics.value())
            writer.add_scalars('valid metric', metrics.value(), step.value)
            checkpoint.update(config.output_dir+'/model.pt', metrics = metrics.value())
            if config.swa and epoch >= int(config.swa_warmup_ratio*config.finetune_epoch):
                metrics_swa = evaluate(model_swa.module, valid_loader)
                logger.log('valid swa', step_swa.value, metrics_swa.value())
                writer.add_scalars('valid metric of swa', metrics_swa.value(), step_swa.value)
                checkpoint_swa.update(config.output_dir+'/model_swa.pt', metrics = metrics_swa.value())

            if config.ema and epoch >= int(config.ema_warmup_ratio*config.finetune_epoch):
                metrics_ema = evaluate(model_ema.module, valid_loader)
                logger.log('valid ema', step_ema.value, metrics_ema.value())
                writer.add_scalars('valid metric of ema', metrics_ema.value(), step_ema.value)
                checkpoint_ema.update(config.output_dir+'/model_ema.pt', metrics = metrics_ema.value())

        checkpoint.save(config.output_dir+'/model_last.pt')
        if config.swa and epoch >= int(config.swa_warmup_ratio*config.finetune_epoch):
            checkpoint_swa.save(config.output_dir+'/model_swa_last.pt')
        if config.ema and epoch >= int(config.ema_warmup_ratio*config.finetune_epoch):
            checkpoint_ema.save(config.output_dir+'/model_ema_last.pt')
    logger.close()
    writer.close()

@torch.no_grad()
def inference(model_file, data_file):
    
    test_data = Sep2SepDataset(data_file, config.input_l, config.output_l,tokenizer=tokenizer,test=True)
    # test_loader = DataLoader(test_data, batch_size=config.valid_batch, shuffle=False, num_workers=1, drop_last=False)
    
    test_chunked_batch_sampler = ChunkedBatchSampler(test_data, config.valid_batch, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_data,  num_workers=1, batch_sampler=test_chunked_batch_sampler)
    model = GenerationModel(config)
    checkpoint = Checkpoint(model = model)
    checkpoint.resume(model_file)
    
    model = model.cuda()    
    model.eval()
    
    fp = open(os.path.join(config.output_dir,'pred.csv'), 'w', newline='')
    writer = csv.writer(fp)
    tot = 0
    for source in tqdm(test_loader):
        source = list(source)
        pred = model(source)
        pred = pred.cpu().numpy()
        pred = tokenizer.batch_decode(pred,skip_special_tokens=True)

        for i in range(len(pred)):
            preds = [tot, pred[i]]
            writer.writerow(preds)
            tot += 1
    fp.close()

config = build_config()

tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
if __name__=='__main__':

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    # only_eval
    if config.do_eval_only:
        only_eval()
    #pretrain
    if config.do_pretrain:
        model_dir = config.output_dir
        pretrain(5)
        model_path_pretrained = os.path.join(config.output_dir , 'model_last.pt')
        config.pretrained_checkpoint = model_path_pretrained
        config.output_dir = model_dir
    #finetune
    if config.do_finetune:
        finetune()
        config.pretrained_checkpoint = os.path.join(config.output_dir , 'model_cider.pt')

    #attack
    if config.do_attack:
        config.PGD = True
        config.FGM = True
        config.AWP = True
        config.Switch = True
        config.output_dir = config.output_dir + '_Switch'
        finetune()
        config.pretrained_checkpoint = os.path.join(config.output_dir , 'model_cider.pt')

    #inference
    if config.do_inference:
        inference(config.pretrained_checkpoint, config.test_file)

