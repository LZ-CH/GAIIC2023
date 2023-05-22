import argparse
import os
def build_config():
    config = argparse.ArgumentParser(description='PyTorch GAIIC')
    
    #base I/O
    config.add_argument('--model_name', type=str,default='fnlp/bart-base-chinese')
    config.add_argument('--tokenizer_name',type=str,default='./tokenizers/bart-base-chinese-gaiic')
    config.add_argument('--train_file',nargs='+',type=str,default=['./gaiic_dataset/semi_train_split.csv',
                                                                   './gaiic_dataset/train.csv'])
    config.add_argument('--valid_file',type=str,default='./gaiic_dataset/semi_valid_split.csv')
    config.add_argument('--test_file',type=str,default='./gaiic_dataset/preliminary_a_test.csv')
    config.add_argument('--pretrain_file_list',nargs='+',type=str,default=['./gaiic_dataset/train.csv',
                                                            './gaiic_dataset/preliminary_a_test.csv',
                                                            './gaiic_dataset/preliminary_b_test.csv',
                                                            './gaiic_dataset/semi_train.csv'])
    config.add_argument('--output_dir',type=str,default='./checkpoint')
    config.add_argument('--exp_name',type=str,default='bart-base-DAE')
    config.add_argument('--input_l',type=int,default=200)
    config.add_argument('--output_l',type=int,default=80)
    config.add_argument('--pretrained_checkpoint',default=None)
    config.add_argument('--max_grad_norm',type=float,default=1.0)
    config.add_argument('--warmup_ratio',type=float,default=0.1)
    config.add_argument('--add_prefix',type=bool,default=True)
    
    #pretrain
    config.add_argument('--pretrain_epoch',type=int,default=100)
    config.add_argument('--classfier_head_warmup_epoch',type=int,default=5)
    
    config.add_argument('--pretrain_lr',type=float,default=5e-5)
    config.add_argument('--accumulation_steps_pretrain',type=int,default=1)
    config.add_argument('--pretrain_batch',type=int,default=64)
    
    
    #finetune
    config.add_argument('--finetune_epoch',type=int,default=10)
    config.add_argument('--finetune_lr',type=float,default=1e-4)
    config.add_argument('--finetune_batch',type=int,default=32)
    config.add_argument('--accumulation_steps',type=int,default=1)
    
    #val
    config.add_argument('--beam',type=int,default=4)
    config.add_argument('--valid_batch',type=int,default=128)
    #task
    config.add_argument('--do_eval_only',action='store_true')
    config.add_argument('--do_pretrain',action='store_true')
    config.add_argument('--do_finetune',action='store_true')
    config.add_argument('--do_inference',action='store_true')
    config.add_argument('--do_attack',action='store_true')
    
    
    #attack
    config.add_argument('--attack_start_epoch',type=int,default=0)
    
    config.add_argument('--PGD',action='store_true')
    config.add_argument('--attack_step',default=3)
    
    config.add_argument('--FGM',action='store_true')
    config.add_argument('--AWP',action='store_true')
    config.add_argument('--Switch',action='store_true')
    config.add_argument('--adv_param',type=str,default='shared')
    #ema
    config.add_argument('--ema',action='store_true')
    config.add_argument('--ema_steps',type=int,default=32)
    config.add_argument('--ema_warmup_ratio',type=float,default = 0.1)
    config.add_argument('--ema_decay',default=0.999)
    
    #swa
    config.add_argument('--swa',action='store_true')
    config.add_argument('--swa_warmup_ratio',type=float,default = 0.5)
    config.add_argument('--swa_lr',default = 1e-6)
    
    #R_drop
    config.add_argument('--R_drop',action='store_true')
    config.add_argument('--alpha',type=float,default=1.0)
    config.add_argument('--simcse_alpha',type=float,default=1.0)
    config.add_argument('--amp',type=bool,default=True)
    config = config.parse_args()
    config.output_dir = os.path.join(config.output_dir, config.exp_name)
    return config
    
    