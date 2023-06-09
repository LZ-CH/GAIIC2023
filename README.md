# 代码说明
[GAIIC2023赛道一](https://www.heywhale.com/org/gaiic2023/competition/area/63fef766b4422ee27402289d/content) Rank-12方案代码。

## 赛题背景
医学影像（如 CT 影像、核磁共振影像）是病情诊断的重要依据，通过医学影像得出诊断报告是针对过程中的重要步骤，也是医疗 AI 研究的前沿热点。本赛道任务要求参赛队伍根据医生对 CT 的影像描述文本数据（即对医学影像特征的描述），生成诊断报告文本。复赛阶段将额外引入临床诊断信息。
与传统文本生成任务不同的是，医学影像诊断报告内容具有专业性、明确性和离散性，因此也需要针对性的算法与模型设计。报告生成结果按照指定评价指标（见提交&评审介绍）进行评测和排名，得分最优者获胜。
## 环境配置

所需环境如下:

python：3.7

torch： 1.12.1+cu113

其它可通过requirements.txt进行配置:
```
pip install -r ./requirements.txt 
```


## 算法

* 加载预训练模型BART/cpt/pegasus，修改原词表，在vocab.txt上更换所需要的词表，三个模型可共用tokenizers/bart-base-chinese-gaiic作为tokenizer来源。
* 对原有模型进行MLM、DAE两种预训练；
* 在训练数据上进行微调10 epoch；
* 在微调基础上进行对抗训练10 epoch，增强其鲁棒性；

使用的tricks:
* 微调阶段: R-drop
* 对抗阶段: PGD、FGM、AWP、Switch(随机对抗)、EMA、SWA、权重平均(对原始参数、EMA参数、SWA参数三者平均)
* 测试阶段: 使用多模型融合，将bart、cpt、pegasus三种模型进行融合，将每一个next_token的logits进行加权平均，并进行结果预测。
* 在整个训练阶段都采用混合精度、BlockShuffle进行加速。
* 数据增广，对decoder的输入进行15%随机替换为其它词，以减少曝光偏差。

## 训练流程
以下为bart单个模型的训练过程
1. 进入工作目录
```
cd GAIIC2023
```
2. 随机数据分割
```
python random_split.py
```
3. 预训练与微调
```
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_pretrain --do_finetune \
--exp_name bart-base-MLM-DAE \
--output_dir ./checkpoint/bart-base \
--model_name fnlp/bart-base-chinese \
--swa --ema --R_drop
```
4. Switch对抗
```
CUDA_VISIBLE_DEVICES=0 python main.py  \
--do_finetune \
--exp_name bart-base-MLM-DAE-Switch \
--output_dir ./checkpoint/bart-base \
--model_name fnlp/bart-base-chinese \
--swa --ema --R_drop \
--Switch --PGD --AWP --FGM \
--pretrained_checkpoint ./checkpoint/bart-base/bart-base-MLM-DAE/model_cider.pt

```
5. 模型权重平均
```
CUDA_VISIBLE_DEVICES=0 python model_swa_creat.py \
--model_path_list ./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_ema_cider.pt \
./checkpoint/bart-base/bart-base-MLM-DAE-Switch/model_swa_cider.pt \
--model_output_path ./checkpoint/bart-base/bart-base-MLM-DAE-Switch/bart-base-switch-avg.pt
```

更多模型训练示例可见train.sh
## 测试流程
在project项目中包含推理过程以及融合代码,融合代码封装在FusionGenerationModel.py中，具体在1008行中对cpt模型的输入参数进行了特殊处理，其他的模型则相差无几

单模测试示例:
```
cd project
python index.py
```
相关融合可见test.sh


## Other
第一次参加NLP的比赛，最终拿到Rank-12的排名，虽然没有拿到奖金，但过程中属实是学到了很多关于NLP的实践知识。cpt的训练是在复赛B榜时才匆忙训练的，导致后续还有2个CPT的模型没有提交上去进行融合，不然极有可能能够进到top-10的哈哈哈。