CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_pretrain --do_finetune \
--exp_name bart-base-MLM-DAE \
--output_dir ./checkpoint/bart-base \
--model_name fnlp/bart-base-chinese \
--swa --ema --R_drop

CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_finetune \
--exp_name bart-base-MLM-DAE-Switch \
--output_dir ./checkpoint/bart-base \
--model_name fnlp/bart-base-chinese \
--swa --ema --R_drop \
--Switch --PGD --AWP --FGM \
--pretrained_checkpoint ./checkpoint/bart-base/bart-base-MLM-DAE/model_cider.pt

CUDA_VISIBLE_DEVICES=0 python model_swa_creat.py \
--model_path_list ./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_ema_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_swa_cider.pt \
--model_output_path ./checkpoint/bart-base/bart-base-MLM-DAE-Switch/bart-base-switch-avg.pt