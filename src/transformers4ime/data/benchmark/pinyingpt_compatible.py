"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""

import torch

from transformers4ime.data.benchmark import Benchmark, register_benchmark
from transformers4ime.data.pinyin import wrap_pinyin_to_tokens
from transformers4ime.utils.logger import LOGGER


@register_benchmark('pinyingpt-compatible')
class PinyinGPTCompatibleBenchmark(Benchmark):

    def construct_context(self, context, pinyin, **kwargs):
        return ''.join(context) + self.tokenizer.sep_token + ''.join(wrap_pinyin_to_tokens(pinyin))

    def inference_sample(self, context, target):
        pinyin = self.get_pinyin(target, self.opts.abbr_mode)

        # context = self.construct_context(context, pinyin)
        # target = ''.join(target)
        context_ids = self.tokenizer.encode_plus(''.join(context))["input_ids"][:-1]
        pinyin_constraint_ids = self.tokenizer.encode_plus(''.join(wrap_pinyin_to_tokens(pinyin)),
                                                           add_special_tokens=False)["input_ids"]

        LOGGER.debug(f"\n Context: {context} \n"
                     f"\n  Context Ids: {context_ids} \n"
                     f"\n  Target: {target} \n"
                     f"\n  Pinyin: {pinyin} \n")

        outputs = self.model.generate(input_ids=torch.Tensor(context_ids).long().unsqueeze(0).to(self.model.device),
                                      constraint_ids=torch.Tensor(pinyin_constraint_ids).long().unsqueeze(0).to(self.model.device),
                                      num_beams=self.opts.num_beams,
                                      num_return_sequences=min(10, self.opts.num_beams),
                                      max_length=len(context_ids) + len(pinyin),
                                      bos_token_id=self.tokenizer.cls_token_id,
                                      eos_token_id=self.tokenizer.pad_token_id,
                                      pad_token_id=self.tokenizer.pad_token_id)

        return outputs[:, len(context_ids):]
