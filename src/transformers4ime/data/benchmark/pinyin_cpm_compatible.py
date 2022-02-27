"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""

import torch

from transformers4ime.data.benchmark import Benchmark, register_benchmark
from transformers4ime.data.pinyin import wrap_pinyin_to_tokens
from transformers4ime.utils.logger import LOGGER


@register_benchmark('pinyin-cpm-compatible')
class PinyinCPMCompatibleBenchmark(Benchmark):

    def construct_context(self, context, pinyin, **kwargs):
        return self.tokenizer.bos_token + ''.join(context) + self.tokenizer.sep_token + ''.join(
            wrap_pinyin_to_tokens(pinyin))

    def inference_sample(self, context, target):
        pinyin = self.get_pinyin(target, self.opts.abbr_mode)

        context = self.construct_context(context, pinyin)
        target = ''.join(target)
        context_ids = self.tokenizer.encode_plus(context)["input_ids"][:-1]

        LOGGER.debug(f"\n Context: {context} \n"
                     f"\n  Context Ids: {context_ids} \n"
                     f"\n  Target: {target} \n"
                     f"\n  Pinyin: {pinyin} \n")

        encoded = context_ids

        outputs = self.model.generate(input_ids=torch.Tensor(encoded).long().unsqueeze(0).to(self.model.device),
                                      num_beams=self.opts.num_beams,
                                      num_return_sequences=min(10, self.opts.num_beams),
                                      max_length=len(encoded) + len(pinyin),
                                      bos_token_id=self.tokenizer.bos_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return outputs[:, len(encoded):]
