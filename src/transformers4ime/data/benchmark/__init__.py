"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""

import importlib
import os
from abc import abstractmethod
from time import time

import torch
from tqdm import tqdm

from transformers4ime.data.pinyin import get_pinyin_with_mode, convert_pinyin_to_ids
from transformers4ime.utils.logger import LOGGER
from transformers4ime.utils.misc import construct_model


class Benchmark:

    def __init__(self, model_cls, opts):
        self.opts = opts

        self.model, self.tokenizer = construct_model(model_cls, opts)
        if opts.best_pt:
            LOGGER.info(f"Loading best checkpoint from: {opts.best_pt}")
            self.model.load_state_dict(torch.load(opts.best_pt), strict=True)

        self.model.cuda(opts.device)

        self.model.eval()

        LOGGER.debug(f"Vocabulary Size: {len(self.tokenizer.get_vocab())}, Tokenizer Len: {len(self.tokenizer)}")

    @staticmethod
    def get_pinyin(words, abbr_mode):
        return get_pinyin_with_mode(words, abbr_mode)

    def build_pinyin_input(self, input_ids, pinyin_mask):
        return [max(0, idx - self.opts.pinyin_start_id) for idx, pm in zip(input_ids, pinyin_mask)]

    def convert_pinyin_to_ids(self, pinyin):
        return convert_pinyin_to_ids(self.tokenizer, pinyin)

    @abstractmethod
    def construct_context(self, context, pinyin, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def inference_sample(self, context, target):
        raise NotImplementedError

    def clean_context(self, context):
        context_tokens = [t if t not in self.opts.additional_special_tokens else self.tokenizer.unk_token for t in
                          self.tokenizer.tokenize(''.join(context))][:512]
        return context_tokens

    def run_eval(self, samples):

        in_top_k = {}
        time_cost = []
        inferences = []
        for sample in tqdm(samples):
            LOGGER.debug(sample)

            context, target = sample
            context = self.clean_context(context)

            idx = None
            start = time()
            try:
                outputs = self.inference_sample(context, target)
                target_str = ''.join(target)
                inference = []
                for i, o in enumerate(outputs):
                    prediction_str = self.tokenizer.decode(o).replace(' ', '')
                    LOGGER.debug(prediction_str)
                    inference.append(prediction_str)
                    if prediction_str == target_str:
                        idx = i
                inferences.append({
                    "context": context,
                    "target": target,
                    "inferences": inference
                })
                time_cost.append(time() - start)
            except Exception as e:
                LOGGER.info([e, context, target])

            for top_k in [1, 5, 10]:
                in_top_k.setdefault(top_k, [])
                score = 1 if idx is not None and idx < top_k else 0
                in_top_k[top_k].append(score)

        avg_time = sum(time_cost) / len(time_cost) * 1000
        LOGGER.info(f"Time Cost: {avg_time} ms")
        return in_top_k, avg_time, inferences


BENCHMARK_REGISTRY = {}


def register_benchmark(name):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_benchmark_cls(cls):
        if name in BENCHMARK_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        BENCHMARK_REGISTRY[name] = cls
        return cls

    return register_benchmark_cls


# automatically import any Python files in the models/ directory
datasets_dir = os.path.dirname(__file__)
for file in os.listdir(datasets_dir):
    path = os.path.join(datasets_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'transformers4ime.data.benchmark.{model_name}')
