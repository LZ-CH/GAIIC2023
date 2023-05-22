---
tags:
- text2text-generation
- Chinese
- seq2seq
- BART
language: zh
---
# Chinese BART-Base

### News

**12/30/2022**

An updated version of CPT & Chinese BART are released. In the new version, we changed the following parts:

- **Vocabulary** We replace the old BERT vocabulary with a larger one of size 51271 built from the training data, in which we 1) add missing 6800+ Chinese characters (most of them are traditional Chinese characters); 2) remove redundant tokens (e.g.  Chinese character tokens with ## prefix); 3) add some English tokens to reduce OOV.
- **Position Embeddings** We extend the max_position_embeddings from 512 to 1024.

We initialize the new version of models with the old version of checkpoints with vocabulary alignment. Token embeddings found in the old checkpoints are copied. And other newly added parameters are randomly initialized. We further train the new CPT & Chinese BART 50K steps with batch size 2048, max-seq-length 1024, peak learning rate 2e-5, and warmup ratio 0.1.

The result compared to the previous checkpoints is as followings:

|            | AFQMC | IFLYTEK | CSL-sum | LCSTS |  AVG  |
| :--------- | :---: | :-----: | :-----: | :---: | :---: |
| Previous   |      |        |        |      |      |
| bart-base  | 73.0 |   60   |  62.1  | 37.8 | 58.23 |
| cpt-base   | 75.1 |  60.5  |  63.0  | 38.2 | 59.20 |
| bart-large | 75.7 |  62.1  |  64.2  | 40.6 | 60.65 |
| cpt-large  | 75.9 |  61.8  |  63.7  | 42.0 | 60.85 |
| Updataed   |      |        |        |      |      |
| bart-base  | 73.03 |  61.25  |  61.51  | 38.78 | 58.64 |
| cpt-base   | 74.40 |  61.23  |  62.09  | 38.81 | 59.13 |
| bart-large | 75.81 |  61.52  |  64.62  | 40.90 | 60.71 |
| cpt-large  | 75.97 |  61.63  |  63.83  | 42.08 | 60.88 |

The result shows that the updated models maintain comparative performance compared with previous checkpoints. There are still some cases that the updated model is slightly worse than the previous one, which results from the following reasons: 1) Training additional a few steps did not lead to significant performance improvement; 2) some downstream tasks are not affected by the newly added tokens and longer encoding sequences, but sensitive to the fine-tuning hyperparameters.

- Note that to use updated models, please update the  `modeling_cpt.py` (new version download [Here](https://github.com/fastnlp/CPT/blob/master/finetune/modeling_cpt.py)) and the vocabulary (refresh the cache).

## Model description

This is an implementation of Chinese BART-Base.

[**CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation**](https://arxiv.org/pdf/2109.05729.pdf)

Yunfan Shao, Zhichao Geng, Yitao Liu, Junqi Dai, Fei Yang, Li Zhe, Hujun Bao, Xipeng Qiu

**Github Link:** https://github.com/fastnlp/CPT

## Usage

```python
>>> from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
>>> tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
>>> model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
>>> text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
>>> text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=False)
    [{'generated_text': '北 京 是 中 国 的 首 都'}]
```

**Note: Please use BertTokenizer for the model vocabulary. DO NOT use original BartTokenizer.**

## Citation

```bibtex
@article{shao2021cpt,
  title={CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation}, 
  author={Yunfan Shao and Zhichao Geng and Yitao Liu and Junqi Dai and Fei Yang and Li Zhe and Hujun Bao and Xipeng Qiu},
  journal={arXiv preprint arXiv:2109.05729},
  year={2021}
}
```
