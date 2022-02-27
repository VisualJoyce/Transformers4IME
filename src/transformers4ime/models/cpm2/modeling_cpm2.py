import math
from typing import Tuple, Callable, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, LogitsProcessorList
from transformers.generation_logits_process import EncoderNoRepeatNGramLogitsProcessor, \
    PrefixConstrainedLogitsProcessor, ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor, \
    InfNanRemoveLogitsProcessor, MinLengthLogitsProcessor, HammingDiversityLogitsProcessor, \
    RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor, NoBadWordsLogitsProcessor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from transformers4ime.data.logits_processor import LOGITS_PROCESSOR_REGISTRY
from transformers4ime.models import register_model
from transformers4ime.models.cpm2.configuration_cpm2 import CPM2Config
from transformers4ime.utils.logger import LOGGER


class MLP(nn.Module):
    def __init__(self, embedding_size):
        super(MLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(embedding_size, embedding_size * 4)
        self.dense_4h_to_h = nn.Linear(embedding_size * 4, embedding_size)
        self.act = nn.functional.gelu

    def forward(self, x):
        h = self.act(self.dense_h_to_4h(x))
        h2 = self.dense_4h_to_h(h)
        return h2


class Attention(nn.Module):
    def __init__(self,
                 embedding_size,
                 num_attention_heads,
                 attention_dropout,
                 residual_dropout):
        super(Attention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.size_per_head = embedding_size // num_attention_heads
        self.embedding_size = embedding_size

        self.query_key_value = nn.Linear(embedding_size, embedding_size * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.resid_drop = nn.Dropout(residual_dropout)
        self.dense = nn.Linear(embedding_size, embedding_size)

    def split_heads(self, x):
        x = x.reshape([-1, self.seq_len, self.num_attention_heads, self.size_per_head])
        return x.permute(0, 2, 1, 3)

    def forward(self, x, kv_cache=None):
        self.seq_len = x.shape[1]
        x = self.query_key_value(x)
        q, k, v = torch.split(x, split_size_or_sections=self.embedding_size, dim=2)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if kv_cache is not None:
            pk, pv = kv_cache[0], kv_cache[1]
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        cached_kv = torch.stack([k, v])

        attn = torch.matmul(q, k.transpose(-1, -2))  # [B, N, L, S]
        attn = attn / math.sqrt(self.size_per_head)

        # [L, S]
        attention_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.float32, device=x.device))
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.seq_len])

        # adding to softmax -> its like removing them entirely
        attn = attn * attention_mask - 10000.0 * (1.0 - attention_mask)
        attn = nn.Softmax(dim=-1)(attn)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)
        # [B, N, L, S] -> [B, L, N, S]
        y = y.permute(0, 2, 1, 3)
        y = torch.reshape(y, [-1, self.seq_len, self.embedding_size])
        y = self.resid_drop(self.dense(y))

        return y, cached_kv


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.input_layernorm = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attention = Attention(config.n_embd,
                                   config.n_head,
                                   config.attn_pdrop,
                                   config.resid_pdrop)
        self.post_attention_layernorm = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config.n_embd)

    def forward(self, x, kv_cache=None):
        attn, cached_kv = self.attention(self.input_layernorm(x), kv_cache=kv_cache)
        x = x + attn
        z = self.post_attention_layernorm(x)
        z = self.mlp(z)
        x = x + z
        return x, cached_kv


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.final_layernorm = nn.LayerNorm(config.n_embd, eps=1e-5)

    def forward(self, x, kv_cache=None):
        cached_kvs = []
        for i, layer in enumerate(self.layers):
            x, cached_kv = layer(
                x,
                kv_cache=kv_cache[i] if kv_cache is not None else None)
            cached_kvs.append(cached_kv)
        x = self.final_layernorm(x)
        return x, torch.stack(cached_kvs)


class CPMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CPM2Config
    # load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    # def _init_weights(self, module):
    #     """Initialize the weights."""
    #     if isinstance(module, (nn.Linear, Conv1D)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)


class CPM2Model(CPMPreTrainedModel):
    def __init__(self, config, opts=None):
        super(CPM2Model, self).__init__(config)
        self.opts = opts

        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.n_positions, config.n_embd)
        self.emb_drop = nn.Dropout(config.embd_pdrop)
        self.transformer = Transformer(config)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            # "position_ids": position_ids,
            # "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
        }

    def forward(
            self,
            input_ids,
            past_key_values=None,
            labels=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].shape[-2]
        position_ids = torch.arange(past_length, input_ids.shape[-1] + past_length, dtype=torch.int64,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # print(position_ids)
        input_ids = self.word_embeddings(input_ids)
        input_ids = self.emb_drop(input_ids + self.position_embeddings(position_ids))
        # print(x)
        hidden_states, cached_kvs = self.transformer(input_ids, past_key_values)
        lm_logits = torch.matmul(hidden_states, self.word_embeddings.weight.transpose(-1, -2))
        # if use_cache:
        #     return input_ids, cached_kvs
        # return input_ids

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + (hidden_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=cached_kvs,
            hidden_states=hidden_states,
            attentions=None,
            cross_attentions=None,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


@register_model('pinyin-cpm-compatible')
class PinyinGPT2CompatibleCPMModel(CPM2Model):

    def __init__(self, config, opts):
        super().__init__(config, opts)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # for inp in input_ids:
        #     LOGGER.debug(self.opts.tokenizer.decode(inp))

        a = (input_ids == self.opts.sep_token_id).long()
        # LOGGER.debug(a)
        input_ids_0 = input_ids.where(a.cumsum(dim=1).cumsum(dim=1) < 1, torch.zeros_like(input_ids))
        # LOGGER.debug(input_ids_0)
        max_len = (input_ids_0 > 0).sum(dim=1).max()
        tar_len = (a.cumsum(dim=1) > 1).sum(dim=1).max() - 1

        # LOGGER.debug((max_len, tar_len))

        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        _, att_len = attention_mask.size()

        # only last token for inputs_ids if past is defined in kwargs
        if past:
            attention_mask = attention_mask[:, :max_len + tar_len]
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            input_ids = input_ids_0[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

        # for inp in input_ids:
        #     LOGGER.debug(self.opts.tokenizer.decode(inp))

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            # "position_ids": position_ids,
            # "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
        }

    def _get_logits_processor(
            self,
            repetition_penalty: float,
            no_repeat_ngram_size: int,
            encoder_no_repeat_ngram_size: int,
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
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
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
        min_length = min_length if min_length is not None else self.config.min_length
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
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())

        processors.append(
            LOGITS_PROCESSOR_REGISTRY['pinyin-cpm-compatible'](self.opts.sep_token_id,
                                                               self.opts.pc_df,
                                                               pad_token_id=self.opts.pad_token_id))
        return processors
