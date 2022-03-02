"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""

import torch
from transformers import LogitsProcessorList

from transformers4ime.data.benchmark import Benchmark, register_benchmark
from transformers4ime.data.logits_processor.pinyingpt_compose import PinyinGPTComposeLogitsProcessor
from transformers4ime.data.pinyin import wrap_pinyin_to_tokens
from transformers4ime.utils.logger import LOGGER


@register_benchmark('pinyingpt-compose')
class PinyinGPTComposeBenchmark(Benchmark):

    def construct_context(self, context, pinyin, **kwargs):
        # add pinyin to context
        return ''.join(context) + self.tokenizer.sep_token + ''.join(wrap_pinyin_to_tokens(pinyin))

    def inference_sample(self, context, target):
        tokenizer = self.tokenizer

        context_tokens = self.tokenizer.tokenize(''.join(context))
        context_pinyin = self.get_pinyin(context_tokens, self.opts.abbr_mode)

        pinyin = self.get_pinyin(target, self.opts.abbr_mode)

        LOGGER.debug(f"\n Context: {context} \n"
                     f"\n  Target: {''.join(target)} \n"
                     f"\n  Pinyin: {pinyin} \n")

        pinyin_ids = self.convert_pinyin_to_ids(context_pinyin + pinyin)
        pinyin_ids = [max(0, idx - self.opts.pinyin_start_id + 1) for idx in pinyin_ids]

        context_ids = self.tokenizer.encode_plus(''.join(context))["input_ids"][:-1]

        pinyin_constraint_ids = self.tokenizer.encode_plus(''.join(wrap_pinyin_to_tokens(pinyin)),
                                                           add_special_tokens=False)["input_ids"]

        processors = LogitsProcessorList()
        processors.append(PinyinGPTComposeLogitsProcessor(self.tokenizer.sep_token_id, self.opts.pc_df))

        outputs = self.model.generate(input_ids=torch.Tensor(context_ids).long().unsqueeze(0).to(self.model.device),
                                      pinyin_ids=torch.Tensor(pinyin_ids).long().unsqueeze(0).to(self.model.device),
                                      constraint_ids=torch.Tensor(pinyin_constraint_ids).long().unsqueeze(0).to(
                                          self.model.device),
                                      num_beams=self.opts.num_beams,
                                      num_return_sequences=min(10, self.opts.num_beams),
                                      max_length=len(context_ids) + len(pinyin),
                                      bos_token_id=tokenizer.cls_token_id,
                                      eos_token_id=tokenizer.pad_token_id,
                                      pad_token_id=tokenizer.pad_token_id,
                                      logits_processor=processors)

        return outputs[:, len(context_ids):]
