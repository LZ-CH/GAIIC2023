# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch.nn as nn
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoConfig,AutoTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from losses import soft_label_loss,FocalLoss,CE,sparse_soft_label_loss
from modeling_cpt import CPTForConditionalGeneration
class GenerationModel(nn.Module):
    def __init__(self, config,load_pretrained=True):
        super().__init__()
        '''
        fnlp/cpt-base
        fnlp/bart-base-chinese
        facebook/bart-base
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.model_config = AutoConfig.from_pretrained(config.model_name)
        self.model_config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model_config.forced_eos_token_id = self.tokenizer.sep_token_id
        self.model_config.eos_token_id = self.tokenizer.sep_token_id
        self.model_config.pad_token_id = self.tokenizer.pad_token_id
        if load_pretrained:
            if 'cpt' in config.model_name:
                self.model = CPTForConditionalGeneration.from_pretrained(config.model_name,config=self.model_config)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name,config=self.model_config)
        else:
            if 'cpt' in config.model_name:
                self.model = CPTForConditionalGeneration(config=self.model_config)
            else:
                self.model = AutoModelForSeq2SeqLM.from_config(config=self.model_config)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        print('Vocab size:',self.tokenizer.vocab_size)
        self.output_l = config.output_l
        self.input_l = config.input_l
        self.beam = config.beam
    def forward(self, inputs, decoder_labels=None,decoder_labels_noisy=None,return_encoder_last_hidden_state=False):

        inputs = self.tokenizer(inputs, max_length = self.input_l,truncation=True,add_special_tokens=True,padding='longest',return_tensors='pt')
        inputs_ids = inputs['input_ids'].cuda()
        # inputs_ids = inputs_ids[:,1:]
        attention_mask = inputs['attention_mask'].cuda()
        # attention_mask = attention_mask[:,1:]
        if decoder_labels is not None:
            decoder_labels = self.tokenizer(decoder_labels, max_length = self.input_l,truncation=True,add_special_tokens=True,padding='longest',return_tensors='pt')
            decoder_input_ids = decoder_labels['input_ids'].cuda()
            # decoder_input_ids = decoder_input_ids[:,1:]
            decoder_attention_mask = decoder_labels['attention_mask'].cuda()
            
            decoder_labels_noisy = self.tokenizer(decoder_labels_noisy, max_length = self.input_l,truncation=True,add_special_tokens=True,padding='longest',return_tensors='pt')
            decoder_input_ids_noisy = decoder_labels_noisy['input_ids'].cuda()
            decoder_attention_mask_noisy = decoder_labels_noisy['attention_mask'].cuda()
            
            out = self.model(input_ids= inputs_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids = decoder_input_ids_noisy,
                             decoder_attention_mask=decoder_attention_mask_noisy
                             ) #[B,S,512]
            pred = out.logits
            loss = soft_label_loss(pred[:, :-1], decoder_input_ids[:, 1:],ignore_index=self.tokenizer.pad_token_id)
            # loss = CE(pred[:, :-1], decoder_input_ids[:, 1:],ignore_index=self.tokenizer.pad_token_id)
            # loss = sparse_soft_label_loss(pred[:, :-1], decoder_input_ids[:, 1:],ignore_index=self.tokenizer.pad_token_id)
            if return_encoder_last_hidden_state:
                return out.encoder_last_hidden_state,pred,loss 
            return pred,loss                 
        else:
            out = self.model.generate(input_ids=inputs_ids,
                                      attention_mask=attention_mask,
                                      max_length=self.output_l,
                                      num_beams=self.beam,
                                      decoder_start_token_id=self.model_config.decoder_start_token_id,
                                      early_stopping=True,
                                      length_penalty=0.9,
                                      )
            # out = out[:,1:]
            return out
class MaskLM(object):
    def __init__(self, tokenizer_path='bart-base-chinese', mlm_probability=0.15):
        self.mlm_probability = mlm_probability
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def torch_mask_tokens(self, inputs: Any, masks: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[
        Any, Any, Any]:
        """
        inputs: token之后的索引
        目前先不考虑随机替换的情况
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masks[masked_indices] = 0
        labels[~masked_indices] = self.tokenizer.pad_token_id  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=105,high=1732,size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, masks, labels

    def torch_mask_tokens_with_pad(self, inputs: Any, mask: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[
        Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        mask[masked_indices] = 0
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, mask

class GenerationModel_Pretrain(nn.Module):
    def __init__(self,config,load_pretrained=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.model_config = AutoConfig.from_pretrained(config.model_name)
        # self.model_config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.diagnosis_token)
        self.model_config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model_config.forced_eos_token_id = self.tokenizer.sep_token_id
        self.model_config.eos_token_id = self.tokenizer.sep_token_id
        self.model_config.pad_token_id = self.tokenizer.pad_token_id
        if load_pretrained:
            if 'cpt' in config.model_name:
                self.model = CPTForConditionalGeneration.from_pretrained(config.model_name,config=self.model_config)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name,config=self.model_config)
        else:
            if 'cpt' in config.model_name:
                self.model = CPTForConditionalGeneration(config=self.model_config)
            else:
                self.model = AutoModelForSeq2SeqLM.from_config(config=self.model_config)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        print('Vocab size:',self.tokenizer.vocab_size)
        self.output_l = config.output_l
        self.input_l = config.input_l
        self.beam = config.beam
        self.lm = MaskLM(config.tokenizer_name)
        if 'final_logits_bias' in self.model.__dict__:
            self.final_logits_bias = True
        else:
            self.final_logits_bias = False
        
    def forward(self,inputs, decoder_labels=None,decoder_labels_noisy=None,return_encoder_last_hidden_state=False):

        inputs = self.tokenizer(inputs, max_length = self.input_l,truncation=True,add_special_tokens=True,padding='longest',return_tensors='pt')
        inputs_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        if decoder_labels is not None:
            decoder_labels = self.tokenizer(decoder_labels, max_length = self.input_l,truncation=True,add_special_tokens=True,padding='longest',return_tensors='pt')
            decoder_input_ids = decoder_labels['input_ids'].cuda()
            # decoder_input_ids = decoder_input_ids[:,1:]
            decoder_attention_mask = decoder_labels['attention_mask'].cuda()
            # total_loss = 0.
            #MLM
            mlm_input_ids, _, lm_label = self.lm.torch_mask_tokens(decoder_input_ids.cpu(), decoder_attention_mask.cpu())
            mlm_input_ids = mlm_input_ids.cuda()
            lm_label = lm_label.cuda().long()

            encoder_output = self.model.get_encoder()(
                mlm_input_ids,
                attention_mask= decoder_attention_mask.cuda(),
                output_hidden_states=True,
                return_dict=True).last_hidden_state
            if self.final_logits_bias:
                encoder_lm_logits = self.model.lm_head(encoder_output) + self.model.final_logits_bias
            else:
                encoder_lm_logits = self.model.lm_head(encoder_output)
            masked_lm_loss = soft_label_loss(encoder_lm_logits, lm_label,ignore_index=self.tokenizer.pad_token_id)
            
            #DAE

            
            decoder_labels_noisy = self.tokenizer(decoder_labels_noisy, max_length = self.input_l,truncation=True,add_special_tokens=True,padding='longest',return_tensors='pt')
            decoder_input_ids_noisy = decoder_labels_noisy['input_ids'].cuda()
            decoder_attention_mask_noisy = decoder_labels_noisy['attention_mask'].cuda()
            
            out = self.model(input_ids= inputs_ids,
                             attention_mask=attention_mask,
                             decoder_input_ids = decoder_input_ids_noisy,
                             decoder_attention_mask=decoder_attention_mask_noisy
                             ) #[B,S,512]
            pred = out.logits
            loss = soft_label_loss(pred[:, :-1], decoder_input_ids[:, 1:],ignore_index=self.tokenizer.pad_token_id)
            # loss = CE(pred[:, :-1], decoder_input_ids[:, 1:],ignore_index=self.tokenizer.pad_token_id)
            total_loss = loss
            total_loss += masked_lm_loss
            if return_encoder_last_hidden_state:
                return out.encoder_last_hidden_state,pred,total_loss 
            return pred, total_loss          
        else:
            out = self.model.generate(input_ids=inputs_ids,
                                      attention_mask=attention_mask,
                                      max_length=self.output_l,
                                      num_beams=1,
                                      decoder_start_token_id=self.model_config.decoder_start_token_id,
                                      early_stopping=True,
                                      length_penalty=0.9,
                                      )
            # out = out[:,1:]
            return out