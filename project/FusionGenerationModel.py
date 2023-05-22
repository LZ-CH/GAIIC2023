import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import copy
import torch
import torch.distributed as dist
from torch import nn
import numpy as np
import json
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    return torch.div(tensor1, tensor2, rounding_mode="floor")
from transformers.utils import ModelOutput, logging
from transformers import Constraint, DisjunctiveConstraint, PhrasalConstraint,GenerationMixin
from transformers import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer,AutoConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from modeling_cpt import CPTForConditionalGeneration
logger = logging.get_logger(__name__)
@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class ContrastiveSearchEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class ContrastiveSearchDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class GreedySearchEncoderDecoderOutput(ModelOutput):
    
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class SampleDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class SampleEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class BeamSampleDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
@dataclass
class BeamSampleEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]
from transformers import BertTokenizer,AutoModelForSeq2SeqLM,AutoTokenizer
from losses import soft_label_loss,comput_R_drop_loss
class FusionGenerationModel(nn.Module,GenerationMixin):
    def __init__(self, tokenizer_name='fnlp/bart-base-chinese',model_name='fnlp/bart-base-chinese',input_l=200,output_l=80,beam=4,load_pretrained=True,model_list_path='model_list.json', WEIGHT=None):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model_config.forced_eos_token_id = self.tokenizer.sep_token_id
        self.model_config.eos_token_id = self.tokenizer.sep_token_id
        self.model_config.pad_token_id = self.tokenizer.pad_token_id
        if load_pretrained:
            if 'cpt' in model_name:
                self.model = CPTForConditionalGeneration.from_pretrained(model_name,config=self.model_config)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name,config=self.model_config)
        else:
            if 'cpt' in model_name:
                self.model = CPTForConditionalGeneration(config=self.model_config)
            else:
                self.model = AutoModelForSeq2SeqLM.from_config(config=self.model_config)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model = self.model.cuda()
        print('Vocab size:',self.tokenizer.vocab_size)
        self.output_l = output_l
        self.input_l = input_l
        self.beam = beam
        self.config = self.model.config
        
        MODEL_LIST = {
            "bart-base-chinese-gaiic": [
                "../checkpoint/bart-base/bart-base-MLM-DAE-Switch/bart-base-switch-avg.pt",
            ],
        }
        self.WEIGHT = None
        self.model_list = self.read_model_list(model_list=MODEL_LIST)
        self.main_input_name = self.model.main_input_name
        self.device = self.model.device
        self.prepare_inputs_for_generation = self.model.prepare_inputs_for_generation
    def forward(self, inputs, decoder_labels=None,decoder_labels_noisy=None):

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
            return pred,loss                 
        else:
            # with torch.no_grad():
            out = self.generate(input_ids=inputs_ids,
                                attention_mask=attention_mask,
                                max_length=self.output_l,
                                num_beams=self.beam,
                                decoder_start_token_id=self.model_config.decoder_start_token_id,
                                early_stopping=True,
                                length_penalty=0.9,
                                )
            # out = out[:,1:]
            return out
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside "
                f"{input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. models with `input_ids` can also make use of `inputs_embeds`
        if self._can_retrieve_inputs_from_name(inputs, "inputs_embeds", model_kwargs):
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. Only encoder-decoder models can have non `input_ids` input format
        if not self.config.is_encoder_decoder and input_name != "input_ids":
            raise ValueError(
                f"If {input_name} is passed as model-specific keyword "
                "input then model has to be an encoder-decoder and not a "
                f"{self.__class__.__name__}."
            )

        # 5. if `inputs` is still None, try to create `input_ids` from BOS token
        if inputs is None:
            inputs = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        return inputs, input_name, model_kwargs
    def _can_retrieve_inputs_from_name(
        self, inputs: Optional[torch.Tensor], name: str, model_kwargs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        If `inputs` is None and `name` is in both forward function and keyword arguments, then inputs can be retrieved
        from name
        """
        can_retrieve_inputs = model_kwargs.get(name, None) is not None and name in set(
            inspect.signature(self.forward).parameters.keys()
        )

        if can_retrieve_inputs and inputs is not None:
            raise ValueError(f"Cannot only pass one of {name} and {self.main_input_name}")

        return can_retrieve_inputs
    def adjust_logits_during_generation(self, logits: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to adjust the logits in the generate method.
        """
        return logits
    def _prepare_input_ids_for_generation(
        self, bos_token_id: Optional[int], encoder_outputs: Optional[ModelOutput]
    ) -> torch.LongTensor:
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")
        return torch.ones((1, 1), dtype=torch.long, device=self.device) * bos_token_id
    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None,encoder:ModelOutput =None
    ) -> Dict[str, Any]:
        # 1. get encoder

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs
    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = None,
    ) -> torch.LongTensor:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            if device is None:
                device = self.device
            return torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("token_type_ids") is not None:
            model_kwargs["token_type_ids"] = model_kwargs["token_type_ids"].repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(expand_size, dim=0)

        if is_encoder_decoder:
            encoder_outputs = model_kwargs.get("encoder_outputs")
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
                expand_size, dim=0
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
    @staticmethod
    def _expand_inputs_for_cpt_generation(##用于处理cpt的model kwargs
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            device = encoder_outputs.last_hidden_state.device
            encoder_outputs["hidden_states"] = tuple(h.index_select(0, expanded_return_idx.to(device)) \
                 for h in encoder_outputs["hidden_states"])
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        past = None
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems
        elif "past_buckets_states" in outputs:
            past = outputs.past_buckets_states

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past = self._convert_to_standard_cache(past, batch_size=batch_size)
        return past
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past
        model_kwargs["past"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs
    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
    def _get_logits_warper(
        self,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        temperature: Optional[float] = None,
        num_beams: Optional[int] = None,
        renormalize_logits: Optional[bool] = None,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # init warp parameters
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        typical_p = typical_p if typical_p is not None else self.config.typical_p
        temperature = temperature if temperature is not None else self.config.temperature
        # instantiate warpers list
        warpers = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if typical_p is not None and typical_p < 1.0:
            warpers.append(TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        # `LogitNormalization` should always be the last logit processor, when present
        if renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers
    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
        exponential_decay_length_penalty: Tuple,
        logits_processor: Optional[LogitsProcessorList],
        renormalize_logits: Optional[bool],
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.config.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        diversity_penalty = diversity_penalty if diversity_penalty is not None else self.config.diversity_penalty
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )
        exponential_decay_length_penalty = (
            exponential_decay_length_penalty
            if exponential_decay_length_penalty is not None
            else self.config.exponential_decay_length_penalty
        )
        suppress_tokens = suppress_tokens if suppress_tokens is not None else self.config.suppress_tokens
        begin_suppress_tokens = (
            begin_suppress_tokens if begin_suppress_tokens is not None else self.config.begin_suppress_tokens
        )
        if forced_decoder_ids is None and hasattr(self.config, "forced_decoder_ids"):
            forced_decoder_ids = self.config.forced_decoder_ids
        # instantiate processors list

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.config.is_encoder_decoder:
                processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > 0:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(exponential_decay_length_penalty, eos_token_id, input_ids_seq_length)
            )
        if suppress_tokens is not None:
            processors.append(SuppressTokensLogitsProcessor(suppress_tokens))
        if begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = begin_index if (input_ids_seq_length > 1 or forced_bos_token_id is None) else begin_index + 1
            if forced_decoder_ids is not None:
                begin_index += forced_decoder_ids[-1][0]  # generation starts after the last token that is forced
            processors.append(SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens, begin_index))
        if forced_decoder_ids is not None:
            processors.append(ForceTokensLogitsProcessor(forced_decoder_ids))
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors
    def _get_stopping_criteria(
        self, max_length: Optional[int], max_time: Optional[float], stopping_criteria: Optional[StoppingCriteriaList]
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria
    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list
    def compute_transition_beam_scores(
        self,
        sequences: torch.Tensor,
        scores: Tuple[torch.Tensor],
        beam_indices: torch.Tensor,
        eos_token_id: int = None,
    ):
        """compute the transition probabilities of sequences given generation
        scores and beam indices"""

        # 1. reshape scores as [vocab_size * batch_size, # generation steps]
        # with batch_size being 2 * vocab_size and # generation steps being
        # seq_len - input_length
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

        # 2. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 3. Set indices of beams that finished early to 0
        # such indices will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # 4. multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * self.config.vocab_size

        # 5. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        # 6. Compute scores
        transition_scores = scores.gather(0, indices)

        # 7. Mask out transition_scores of beams that stopped early
        transition_scores[beam_indices_mask] = 0

        return transition_scores
    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        if not hasattr(self, "prepare_inputs_for_generation"):
            generate_compatible_mappings = [
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                MODEL_FOR_VISION_2_SEQ_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            raise TypeError(exception_message)
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[int, float]] = None,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        **model_kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # # 0. Validate the `.generate()` call
        # self._validate_model_class()
        # self._validate_model_kwargs(model_kwargs.copy())
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        model_kwargs_ = model_kwargs
        model_kwargs_list = []
        for model in self.model_list:

            inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, copy.deepcopy(model_kwargs_))
            batch_size = inputs_tensor.shape[0]

            # 3. Define other model kwargs
            model_kwargs["output_attentions"] = output_attentions
            model_kwargs["output_hidden_states"] = output_hidden_states
            model_kwargs["use_cache"] = use_cache

            accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
            requires_attention_mask = "encoder_outputs" not in model_kwargs

            if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
                model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                    inputs_tensor, pad_token_id, eos_token_id
                )

            # decoder-only models should use left-padding for generation
            if not self.config.is_encoder_decoder:
                if pad_token_id is not None and torch.sum(inputs_tensor[:, -1] == pad_token_id) > 0:
                    logger.warning(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

            if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
                # if model is encoder decoder encoder_outputs are created
                # and added to `model_kwargs`
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, model_kwargs, model_input_name,model.get_encoder()
                )
            model_kwargs_list.append(model_kwargs)


        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
                device=inputs_tensor.device,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        if max_length is None and max_new_tokens is None:
            warnings.warn(
                "Neither `max_length` nor `max_new_tokens` has been set, `max_length` will default to "
                f"{self.config.max_length} (`self.config.max_length`). Controlling `max_length` via the config is "
                "deprecated and `max_length` will be removed from the config in v5 of Transformers -- we recommend "
                "using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            raise ValueError(
                "Both `max_new_tokens` and `max_length` have been set but they serve the same purpose -- setting a"
                " limit to the generated output length. Remove one of those arguments. Please refer to the"
                " documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length

        if min_length is not None and min_length > max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({min_length}) is larger than the maximum "
                f"length ({max_length})"
            )
        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {max_length}. This can lead to unexpected behavior. You should consider increasing "
                "`max_new_tokens`."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None

        is_contrastive_search_gen_mode = (
            top_k is not None and top_k > 1 and do_sample is False and penalty_alpha is not None and penalty_alpha > 0
        )

        is_beam_gen_mode = (
            (num_beams > 1)
            and (num_beam_groups == 1)
            and do_sample is False
            and not is_constraint_gen_mode
            and not is_contrastive_search_gen_mode
        )

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
            renormalize_logits=renormalize_logits,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            forced_decoder_ids=forced_decoder_ids,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )
        # 9. go into different generation modes
        
        if is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=inputs_tensor.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            for i,model_kwargs in enumerate(model_kwargs_list):
                if self.model_list[i].config.decoder_layers != self.model_list[i].config.encoder_layers:
                    input_ids_, model_kwargs_list[i] = self._expand_inputs_for_cpt_generation(
                        input_ids=copy.deepcopy(input_ids).cuda(),
                        expand_size=num_beams,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                        **model_kwargs,
                    )
                else:
                    input_ids_, model_kwargs_list[i] = self._expand_inputs_for_generation(
                        input_ids=copy.deepcopy(input_ids).cuda(),
                        expand_size=num_beams,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                        **model_kwargs,
                    )
            input_ids = input_ids_
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                model_kwargs_list = model_kwargs_list,
            )

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        model_kwargs_list:list = None,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        model_list = self.model_list
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None


        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float).cuda()
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            
            models = []
            # next_token_scores_average = torch.zeros((batch_beam_size, self.self.vocab_size())).cuda()
            next_token_logits_average = torch.zeros((batch_beam_size, self.tokenizer.vocab_size)).cuda()
            if(len(self.WEIGHT)!=len(model_list)):
                raise ValueError(
                f"WEIGTH length should be {len(model_list)}, but is {len(self.WEIGHT)}."
            )
            for model,model_kwargs,weight in zip(model_list,model_kwargs_list,self.WEIGHT):
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                models.append(outputs)
                next_token_logits = outputs.logits[:, -1, :]
                # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
                # cannot be generated both before and after the `nn.functional.log_softmax` operation.
                next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
                next_token_logits_average += torch.mul(next_token_logits,weight)

                
            next_token_scores = nn.functional.log_softmax(
                    next_token_logits_average, dim=-1
                    )  # (batch_size * num_beams, vocab_size)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            for i,output,model_kwargs in zip(range(len(model_kwargs_list)),models,model_kwargs_list):
                model_kwargs = self._update_model_kwargs_for_generation(
                    output, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )
                if model_kwargs["past"] is not None:
                    model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)
                model_kwargs_list[i] = model_kwargs

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        return sequence_outputs["sequences"]

    def read_model_list(self,model_list=None):
        
        # 加载模型和tokenizer
        models = []
        for model_name, checkpoint_paths in model_list.items():
            for checkpoint_path in checkpoint_paths:
                # path = os.path.join(os.getcwd(), checkpoint_path)
                config = AutoConfig.from_pretrained(model_name)
                
                config.decoder_start_token_id = self.tokenizer.bos_token_id
                config.forced_eos_token_id = self.tokenizer.sep_token_id
                config.eos_token_id = self.tokenizer.sep_token_id
                config.pad_token_id = self.tokenizer.pad_token_id
                
                if 'cpt' in model_name:
                    model = CPTForConditionalGeneration(config=config)
                else:
                    model = AutoModelForSeq2SeqLM.from_config(config=config)
                model.resize_token_embeddings(self.tokenizer.vocab_size)
                
                checkpoint = torch.load(checkpoint_path)
                print(os.path.basename(checkpoint_path) ,checkpoint['best_metrics'])
                new_checkpoint = {}
                for k in checkpoint['model']:
                    new_checkpoint[k.replace('model.','',1)] = checkpoint['model'][k]
                
                model.load_state_dict(new_checkpoint)
                
                models.append(model.half().cuda())
        if self.WEIGHT is None:
            self.WEIGHT = [1/len(models)]*len(models)
        return models