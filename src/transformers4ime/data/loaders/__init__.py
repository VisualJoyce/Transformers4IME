"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import importlib
import logging
import os
import tarfile
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from itertools import chain, islice

import numpy as np
import torch
from transformers import BatchFeature, PretrainedConfig

from transformers4ime.data.arguments import MMTrainingArguments, MMDataTrainingArguments, MMModelArguments

logger = logging.getLogger(__name__)


def move_to_cuda(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True, device=device)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t, device=device) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t, device=device) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
    elif isinstance(batch, BatchFeature):
        new_batch = {n: move_to_cuda(t, device=device) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch, device):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream(device=device))
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t, device)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t, device)
    else:
        pass


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch, device=self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch, self.device)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


LOADER_REGISTRY = {}


def register_loader(name):
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

    def register_loader_cls(cls):
        if name in LOADER_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        LOADER_REGISTRY[name] = cls
        return cls

    return register_loader_cls


EVAL_LOADER_REGISTRY = {}


def register_eval_loader(name):
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

    def register_eval_loader_cls(cls):
        if name in EVAL_LOADER_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        EVAL_LOADER_REGISTRY[name] = cls
        return cls

    return register_eval_loader_cls


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
        module = importlib.import_module(f'transformers4ime.data.loaders.{model_name}')


class MMLoader(metaclass=ABCMeta):

    def __init__(self, tokenizer,
                 model_args: MMModelArguments,
                 training_args: MMTrainingArguments,
                 data_args: MMDataTrainingArguments,
                 config: PretrainedConfig):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.config = config

        self.all_epochs = defaultdict(int)
        self.step = 0

    @property
    def modalities(self):
        return sorted(
            list([m for m in LOADER_REGISTRY.keys() if
                  getattr(self.data_args, f'train_{m}_files') not in (None, '')]))

    def _use_at_most_k_wrapper(self, loader):
        if self.training_args.use_at_most_k:
            k = self.training_args.use_at_most_k
            return islice(loader, k)
        return loader

    @abstractmethod
    def _interchange_iter(self):
        raise NotImplementedError

    def __iter__(self):
        logging.info(f"Modalities: {self.modalities}")
        while True:
            # self.all_epochs['epoch'] += 1
            # try:
                for batch in PrefetchLoader(self._interchange_iter(), self.training_args.device):
                    yield batch
                    self.step += 1
            # except tarfile.ReadError as e:
            #     logger.warning(e)
            # except ValueError as ve:
            #     logger.warning(ve)


class MMModalityWiseLoader(MMLoader):

    def __init__(self, tokenizer, model_args: MMModelArguments, training_args: MMTrainingArguments,
                 data_args: MMDataTrainingArguments, config: PretrainedConfig):
        super().__init__(tokenizer, model_args, training_args, data_args, config)

    def _roll_modalities(self):
        idx = np.arange(len(self.modalities))
        return [self.modalities[i] for i in np.roll(idx, self.training_args.local_rank)]

    def _interchange_iter(self):
        modalities = self._roll_modalities() if self.training_args.roll_modalities else self.modalities
        loaders = map(self._use_at_most_k_wrapper, [LOADER_REGISTRY[m](self.tokenizer,
                                                                       self.model_args,
                                                                       self.training_args,
                                                                       self.data_args,
                                                                       self.config) for m in modalities])

        for batch in chain(*loaders):
            _check_ = (self.step + 1) % self.training_args.gradient_accumulation_steps == 0
            batch['should_grad_sync_and_apply'] = True if _check_ else False
            batch['gradient_accumulation_steps'] = self.training_args.gradient_accumulation_steps
            yield batch


class MMStepWiseLoader(MMLoader):

    def __init__(self, tokenizer, model_args: MMModelArguments, training_args: MMTrainingArguments,
                 data_args: MMDataTrainingArguments, config: PretrainedConfig):
        super().__init__(tokenizer, model_args, training_args, data_args, config)
        assert self.training_args.roll_modalities is False

        self.all_gradient_accumulation_steps = {
            m: getattr(self.training_args,
                       f'{m}_gradient_accumulation_steps') or self.training_args.gradient_accumulation_steps
            for m in self.modalities
        }
        logger.info(f"all_gradient_accumulation_steps: {self.all_gradient_accumulation_steps}")

        self.interchange_steps = sum(self.all_gradient_accumulation_steps.values())
        logger.info(f"interchange_steps: {self.interchange_steps}")

    def _get_modality_from_step(self):
        step_idx = self.step % self.interchange_steps
        start = 0
        for m in self.modalities:
            gradient_accumulation_steps = self.all_gradient_accumulation_steps[m]
            end = start + gradient_accumulation_steps
            if start <= step_idx < end:
                if step_idx + 1 == end:
                    return m, True, gradient_accumulation_steps
                else:
                    return m, False, gradient_accumulation_steps
            start = end

    def _interchange_iter(self):

        loaders = {m: iter(self._use_at_most_k_wrapper(LOADER_REGISTRY[m](self.tokenizer,
                                                                          self.model_args,
                                                                          self.training_args,
                                                                          self.data_args,
                                                                          self.config))) for m in self.modalities}

        while True:
            m, should_grad_sync_and_apply, gradient_accumulation_steps = self._get_modality_from_step()
            try:
                batch = next(loaders[m])
                batch['should_grad_sync_and_apply'] = should_grad_sync_and_apply
                batch['gradient_accumulation_steps'] = gradient_accumulation_steps
                yield batch
            except StopIteration:
                self.all_epochs[m] += 1
                loaders[m] = iter(self._use_at_most_k_wrapper(LOADER_REGISTRY[m](self.tokenizer,
                                                                                 self.model_args,
                                                                                 self.training_args,
                                                                                 self.data_args,
                                                                                 self.config)))


MM_LOADERS = {
    'step_wise': MMStepWiseLoader,
    'modality_wise': MMModalityWiseLoader
}
