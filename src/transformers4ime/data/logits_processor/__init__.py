import importlib
import os

import torch
from transformers import LogitsProcessor


class ConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, sep_token_id, pc_df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sep_token_id = sep_token_id
        self.pc_df = pc_df

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, constraint_id) -> torch.FloatTensor:
        return scores


LOGITS_PROCESSOR_REGISTRY = {}


def register_logits_processor(name):
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

    def register_logits_processor_cls(cls):
        if name in LOGITS_PROCESSOR_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        LOGITS_PROCESSOR_REGISTRY[name] = cls
        return cls

    return register_logits_processor_cls


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
        module = importlib.import_module(f'transformers4ime.data.logits_processor.{model_name}')
