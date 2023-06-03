#Bart
#Bart-pretrain finetune
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_pretrain --do_finetune \
--exp_name bart-base-MLM-DAE \
--output_dir ./checkpoint/bart-base \
--model_name fnlp/bart-base-chinese \
--swa --ema --R_drop

#Attack-Switch
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_finetune \
--exp_name bart-base-MLM-DAE-Switch \
--output_dir ./checkpoint/bart-base \
--model_name fnlp/bart-base-chinese \
--swa --ema --R_drop \
--Switch --PGD --AWP --FGM \
--pretrained_checkpoint ./checkpoint/bart-base/bart-base-MLM-DAE/model_cider.pt

#Attack-PGD
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_finetune \
--exp_name bart-base-MLM-DAE-PGD \
--output_dir ./checkpoint/bart-base \
--model_name fnlp/bart-base-chinese \
--swa --ema --R_drop \
--Switch --PGD --AWP --FGM \
--pretrained_checkpoint ./checkpoint/bart-base/bart-base-MLM-DAE/model_cider.pt

# CPT
#cpt-pretrain finetune
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_pretrain --do_finetune \
--exp_name cpt-base-MLM-DAE \
--output_dir ./checkpoint/cpt-base \
--model_name fnlp/cpt-base \
--swa --ema --R_drop

#Attack-Switch
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_finetune \
--exp_name cpt-base-MLM-DAE-Switch \
--output_dir ./checkpoint/cpt-base \
--model_name fnlp/cpt-base \
--swa --ema --R_drop \
--Switch --PGD --AWP --FGM \
--pretrained_checkpoint ./checkpoint/cpt-base/cpt-base-MLM-DAE/model_cider.pt

#Attack-PGD
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_finetune \
--exp_name cpt-base-MLM-DAE-PGD \
--output_dir ./checkpoint/cpt-base \
--model_name fnlp/cpt-base \
--swa --ema --R_drop \
--Switch --PGD --AWP --FGM \
--pretrained_checkpoint ./checkpoint/cpt-base/cpt-base-MLM-DAE/model_cider.pt

#pegasus
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_pretrain --do_finetune \
--exp_name pegasus-base-MLM-DAE \
--output_dir ./checkpoint/pegasus-base \
--model_name IDEA-CCNL/Randeng-Pegasus-238M-Chinese \
--swa --ema --R_drop

#Attack-Switch
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_finetune \
--exp_name pegasus-base-MLM-DAE-Switch \
--output_dir ./checkpoint/pegasus-base \
--model_name IDEA-CCNL/Randeng-Pegasus-238M-Chinese \
--swa --ema --R_drop \
--Switch --PGD --AWP --FGM \
--pretrained_checkpoint ./checkpoint/pegasus-base/pegasus-base-MLM-DAE/model_cider.pt

#model avg

##bart-switch
CUDA_VISIBLE_DEVICES=0 python model_swa_creat.py \
--model_path_list ./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_ema_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_swa_cider.pt \
--model_output_path ./checkpoint/bart-base/bart-base-MLM-DAE-Switch/bart-base-switch-avg.pt

##bart-PGD
CUDA_VISIBLE_DEVICES=0 python model_swa_creat.py \
--model_path_list ./checkpoint/bart-base/bart-base-MLM-DAE-PGD/model_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-PGD/model_ema_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-PGD/model_swa_cider.pt \
--model_output_path ./checkpoint/bart-base/bart-base-MLM-DAE-PGD/bart-base-switch-avg.pt

##cpt-switch
CUDA_VISIBLE_DEVICES=0 python model_swa_creat.py \
--model_path_list ./checkpoint/cpt-base/cpt-base-MLM-DAE-Switch/model_cider.pt \
./checkpoint/cpt-base/cpt-base-MLM-DAE-Switch/model_ema_cider.pt \
./checkpoint/cpt-base/cpt-base-MLM-DAE-Switch/model_swa_cider.pt \
--model_output_path ./checkpoint/cpt-base/cpt-base-MLM-DAE-Switch/cpt-base-switch-avg.pt

##cpt-PGD
CUDA_VISIBLE_DEVICES=0 python model_swa_creat.py \
--model_path_list ./checkpoint/cpt-base/cpt-base-MLM-DAE-PGD/model_cider.pt \
./checkpoint/cpt-base/cpt-base-MLM-DAE-PGD/model_ema_cider.pt \
./checkpoint/cpt-base/cpt-base-MLM-DAE-PGD/model_swa_cider.pt \
--model_output_path ./checkpoint/cpt-base/cpt-base-MLM-DAE-Switch/cpt-base-switch-avg.pt

##pegasus-switch
CUDA_VISIBLE_DEVICES=0 python model_swa_creat.py \
--model_path_list ./checkpoint/pegasus-base/pegasus-base-MLM-DAE-Switch/model_cider.pt \
./checkpoint/pegasus-base/pegasus-base-MLM-DAE-Switch/model_ema_cider.pt \
./checkpoint/pegasus-base/pegasus-base-MLM-DAE-Switch/model_swa_cider.pt \
--model_output_path ./checkpoint/pegasus-base/pegasus-base-MLM-DAE-Switch/pegasus-base-switch-avg.pt