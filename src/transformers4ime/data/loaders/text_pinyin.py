"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import collections
import logging
import math
import random

import webdataset as wds
from more_itertools import flatten, unzip
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig

from transformers4ime.data.arguments import MMModelArguments, MMTrainingArguments, MMDataTrainingArguments
from transformers4ime.data.loaders import register_loader
from transformers4ime.data.loaders.base import IMEBaseDataLoader

logger = logging.getLogger(__name__)


@register_loader("text_pinyin")
class IMETextDataLoader(IMEBaseDataLoader):
    def __init__(self, tokenizer, model_args: MMModelArguments, training_args: MMTrainingArguments,
                 data_args: MMDataTrainingArguments, config: PretrainedConfig):
        super().__init__(tokenizer, model_args, training_args, data_args, config)
        self.shards = self.get_shards(self.data_args.train_text_only_files)
        self.batch_size = self.training_args.text_only_per_device_train_batch_size
        self.max_len = self.data_args.text_only_block_size

    def build_pinyin(self, example):

        pivot = random.randint(0, len(len_) - 1)

        prob = random.random()
        # we use 1, 2, 3 words as target randomly
        if prob < 0.5:
            target_len = math.ceil(prob / 0.9 * 10)
        else:
            target_len = random.randint(6, 25)

        if len(len_) < pivot + target_len:
            target_len = len(len_) - pivot + 1

        while sum(len_[pivot:pivot + target_len]) * 2 > opts.max_txt_len:
            target_len -= 1

        pre_context_len = opts.max_txt_len - sum(len_[pivot:pivot + target_len]) * 2 - sum(len_[:pivot])
        if pre_context_len < 0:
            return (key, (mid, mid), (pivot, target_len)), opts.max_txt_len + n_special_tokens

        # append sentence util pre_txt_len is 0
        start = mid
        while pre_context_len > 0 and start > 0:
            start -= 1
            last_len = sum(len_list[start])
            if pre_context_len - last_len < 0:
                pre_context_len = 0
            else:
                pre_context_len -= last_len
        # new id and len
        return self._flatten_mius(pinyin[start:end]) + self._flatten_mius([pinyin[end][:pivot]]), list(
            flatten(pinyin[end][pivot:pivot + target_len]))

    def build_sample(self, example):
        pre_pinyin_ids, post_pinyin_ids = self.build_pinyin(example)

        tokens = example[0]
        post_context_ids = list(flatten(tokens[end][pivot:pivot + target_len]))
        pre_context_ids = self._flatten_mius(tokens[start:end]) + self._flatten_mius([tokens[end][:pivot]])
        if len(pre_context_ids) > self.config.max_txt_len - len(post_pinyin_ids):
            pre_context_len = len(pre_context_ids) - (self.config.max_txt_len - len(post_pinyin_ids))
            pre_context_ids = pre_context_ids[pre_context_len:]
            pre_pinyin_ids = pre_pinyin_ids[pre_context_len:]

        context_ids = pre_context_ids + post_context_ids
        pinyin_ids = pre_pinyin_ids + post_pinyin_ids
        assert len(context_ids) == len(pinyin_ids)

        pinyin_ids = [max(0, idx - self.pinyin_start_id + 1) for idx in pinyin_ids]
        label_ids = [idx if pid > 0 else -100 for idx, pid in zip(context_ids, pinyin_ids)]

        position_ids = list(range(1, 1 + len(context_ids)))

        input_ids = [self.tokenizer.cls_token_id] + context_ids + [self.tokenizer.sep_token_id]
        label_ids = [-100] + label_ids + [-100]
        pinyin_ids = pinyin_ids + [0] + [0]
        position_ids = [0] + position_ids + [len(label_ids) - 1]
        attention_mask = [1] * len(input_ids)
        return context_ids, position_ids, label_ids, pinyin_ids

    @staticmethod
    def collate_fn(inputs):
        (input_ids, attention_mask, position_ids, label_ids, pinyin_ids) = map(list,
                                                                               unzip([item for item in inputs if
                                                                                      None not in item]))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
        pinyin_ids = pad_sequence(pinyin_ids, batch_first=True, padding_value=0)

        batch = {'input_ids': input_ids,
                 # 'token_type_ids': token_type_ids,
                 'attention_mask': attn_masks,
                 'pinyin_ids': pinyin_ids,
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
                .to_tuple("json")
        )
        for d in dataset.batched(self.batch_size):
            yield self.convert_to_features(d)
