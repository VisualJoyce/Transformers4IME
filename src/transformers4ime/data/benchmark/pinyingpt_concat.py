"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""

import torch

from transformers4ime.data.benchmark import Benchmark, register_benchmark
from transformers4ime.data.pinyin import wrap_pinyin_to_tokens
from transformers4ime.utils.logger import LOGGER


@register_benchmark('pinyingpt-concat')
class PinyinGPTConcatBenchmark(Benchmark):

    def construct_context(self, context, pinyin, **kwargs):
        concat_mode = kwargs['concat_mode']
        if concat_mode == 'segmented':
            return ''.join(context) + self.tokenizer.sep_token + ''.join(wrap_pinyin_to_tokens(pinyin))
        else:
            return ''.join(context) + ''.join(wrap_pinyin_to_tokens(pinyin))

    def inference_sample_finetune(self, context, target):
        pinyin = self.get_pinyin(target, self.opts.abbr_mode)

        context = self.construct_context(context, pinyin, concat_mode=self.opts.concat_mode)

        LOGGER.debug(f"\n Context: {context} \n"
                     f"\n  Target: {target} \n"
                     f"\n  Pinyin: {pinyin} \n")
        context_ids = self.tokenizer.encode_plus(context)["input_ids"]
        pinyin_constraint_ids = self.tokenizer.encode_plus(''.join(wrap_pinyin_to_tokens(pinyin)),
                                                           add_special_tokens=False)["input_ids"]

        context_ids = context_ids if self.opts.concat_mode == 'segmented' else context_ids[:-1]  # drop [SEP]

        if self.opts.position_mode == 'positional':
            position_ids = None
        else:
            sep_idx = context_ids.index(self.tokenizer.sep_token_id)
            context_position_ids = list(range(sep_idx + 1))
            pinyin_position_ids = list(range(sep_idx + 1, sep_idx + 1 + len(pinyin) + 1))
            position_ids = context_position_ids + pinyin_position_ids + pinyin_position_ids
            # assert len(context_ids) == len(position_ids)
            position_ids = torch.Tensor(position_ids).long().unsqueeze(0).to(self.model.device)

        outputs = self.model.generate(input_ids=torch.Tensor(context_ids).long().unsqueeze(0).to(self.model.device),
                                      position_ids=position_ids,
                                      constraint_ids=torch.Tensor(pinyin_constraint_ids).long().unsqueeze(0).to(self.model.device),
                                      num_beams=self.opts.num_beams,
                                      num_return_sequences=min(10, self.opts.num_beams),
                                      max_length=len(context_ids) + len(pinyin),
                                      bos_token_id=self.tokenizer.cls_token_id,
                                      eos_token_id=self.tokenizer.pad_token_id,
                                      pad_token_id=self.tokenizer.pad_token_id)

        return outputs[:, len(context_ids):]

    def inference_sample_fixed(self, context, target):
        pinyin = self.get_pinyin(target, self.opts.abbr_mode)
        pinyin_ids = self.convert_pinyin_to_ids(pinyin)

        LOGGER.debug(f"\n Context: {context} \n"
                     f"\n  Target: {target} \n"
                     f"\n  Pinyin: {pinyin} \n")

        context_ids = self.tokenizer.encode_plus(''.join(context))["input_ids"]

        pinyin_masks = [0] * len(context_ids) + [1] * len(pinyin) + [0]
        context_ids = context_ids + pinyin_ids + [self.tokenizer.sep_token_id]
        context_ids = context_ids if self.opts.concat_mode == 'segmented' else context_ids[:-1]  # drop [SEP]

        pinyin_ids = self.build_pinyin_input(context_ids, pinyin_masks)
        if self.opts.position_mode == 'positional':
            position_ids = None
        else:
            sep_idx = context_ids.index(self.tokenizer.sep_token_id)
            context_position_ids = list(range(sep_idx + 1))
            pinyin_position_ids = list(range(sep_idx + 1, sep_idx + 1 + len(pinyin) + 1))
            position_ids = context_position_ids + pinyin_position_ids + pinyin_position_ids
            position_ids = torch.Tensor(position_ids).long().unsqueeze(0).to(self.model.device)

        pinyin_constraint_ids = self.tokenizer.encode_plus(''.join(wrap_pinyin_to_tokens(pinyin)),
                                                           add_special_tokens=False)["input_ids"]
        outputs = self.model.generate(input_ids=torch.Tensor(context_ids).long().unsqueeze(0).to(self.model.device),
                                      position_ids=position_ids,
                                      pinyin_ids=torch.Tensor(pinyin_ids).long().unsqueeze(0).to(self.model.device),
                                      pinyin_masks=torch.Tensor(pinyin_masks).long().unsqueeze(0).to(self.model.device),
                                      constraint_ids=torch.Tensor(pinyin_constraint_ids).long().unsqueeze(0).to(self.model.device),
                                      num_beams=self.opts.num_beams,
                                      num_return_sequences=min(10, self.opts.num_beams),
                                      max_length=len(context_ids) + len(pinyin),
                                      bos_token_id=self.tokenizer.cls_token_id,
                                      eos_token_id=self.tokenizer.pad_token_id,
                                      pad_token_id=self.tokenizer.pad_token_id)

        return outputs[:, len(context_ids):]

    def inference_sample(self, context, target):
        if self.opts.gpt2_fixed:
            return self.inference_sample_fixed(context, target)
        else:
            return self.inference_sample_finetune(context, target)
