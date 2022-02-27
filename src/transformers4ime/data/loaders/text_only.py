"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import collections
import logging

import torch
import webdataset as wds
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig

from transformers4ime.data.arguments import MMModelArguments, MMTrainingArguments, MMDataTrainingArguments
from transformers4ime.data.loaders import register_loader
from transformers4ime.data.loaders.base import IMEBaseDataLoader

logger = logging.getLogger(__name__)


@register_loader("text_only")
class IMETextOnlyDataLoader(IMEBaseDataLoader):
    def __init__(self, tokenizer, model_args: MMModelArguments, training_args: MMTrainingArguments,
                 data_args: MMDataTrainingArguments, config: PretrainedConfig):
        super().__init__(tokenizer, model_args, training_args, data_args, config)
        self.shards = self.get_shards(self.data_args.train_text_only_files)
        self.batch_size = self.training_args.text_only_per_device_train_batch_size
        self.max_len = self.data_args.text_only_block_size
        self.n_ctx = self.config.n_ctx

    def build_sample(self, example):
        context_ids = self.tokenizer(example['content'], add_special_tokens=False)['input_ids']
        if len(context_ids) + 2 > self.n_ctx:
            context_ids = context_ids[:self.n_ctx - 2]

        label_ids = [idx if idx > 0 else -100 for idx in context_ids]

        position_ids = list(range(1, 1 + len(context_ids)))

        input_ids = [self.tokenizer.cls_token_id] + context_ids + [self.tokenizer.sep_token_id]
        label_ids = [-100] + label_ids + [-100]
        position_ids = [0] + position_ids + [len(label_ids) - 1]
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids)
        label_ids = torch.tensor(label_ids)
        position_ids = torch.tensor(position_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, attention_mask, position_ids, label_ids

    @staticmethod
    def collate_fn(inputs):
        (input_ids, attention_mask, position_ids, label_ids) = map(list,
                                                                   unzip([item for item in inputs if
                                                                          None not in item]))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)

        batch = {'input_ids': input_ids,
                 # 'token_type_ids': token_type_ids,
                 'attention_mask': attn_masks,
                 # 'gather_index': gather_index,
                 'position_ids': position_ids,
                 # 'option_ids': torch.tensor(options).long(),
                 'labels': label_ids}
        return batch

    def convert_to_features(self, data):
        d_lists = collections.defaultdict(list)
        # truth_ids, input_ids, segment_ids, attention_masks = [], [], [], []
        inputs = [self.build_sample(item) for item in data]
        return self.collate_fn(inputs)

    def __iter__(self):
        assert len(self.shards) >= self.training_args.world_size  # guarantee at least one shard for each device
        logging.info(f"Constructing data loader for image text: {len(self.shards)}")
        dataset = (
            wds.WebDataset(self.shards)
                .shuffle(1000)
                .decode()
                .to_tuple("json", )
        )
        for d, in dataset.batched(self.batch_size):
            yield self.convert_to_features(d)
