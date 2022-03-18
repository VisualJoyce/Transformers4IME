"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import collections
import json
import logging
import math
import random
import re

import pandas as pda
import torch
import webdataset as wds
from more_itertools import unzip, flatten
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PretrainedConfig

from transformers4ime.data.arguments import MMModelArguments, MMTrainingArguments, MMDataTrainingArguments
from transformers4ime.data.loaders import register_loader
from transformers4ime.data.loaders.base import IMEBaseDataLoader
from transformers4ime.data.pinyin import get_pinyin_to_char, convert_pinyin_to_ids, get_pinyin_with_mode

logger = logging.getLogger(__name__)

Tagger = collections.namedtuple('Tagger', 'cut')

n2l = {2: 'abc', 3: 'def', 4: 'ghi', 5: 'jkl', 6: 'mno', 7: 'pqrs', 8: 'tuv', 9: 'wxyz'}
l2n = {'a': 2,
       'b': 2,
       'c': 2,
       'd': 3,
       'e': 3,
       'f': 3,
       'g': 4,
       'h': 4,
       'i': 4,
       'j': 5,
       'k': 5,
       'l': 5,
       'm': 6,
       'n': 6,
       'o': 6,
       'p': 7,
       'q': 7,
       'r': 7,
       's': 7,
       't': 8,
       'u': 8,
       'v': 8,
       'w': 9,
       'x': 9,
       'y': 9,
       'z': 9}


@register_loader("text_pinyin")
class IMETextPinyinDataLoader(IMEBaseDataLoader):
    def __init__(self, tokenizer, model_args: MMModelArguments, training_args: MMTrainingArguments,
                 data_args: MMDataTrainingArguments, config: PretrainedConfig):
        super().__init__(tokenizer, model_args, training_args, data_args, config)
        self.shards = self.get_shards(self.data_args.train_text_pinyin_files)
        self.batch_size = self.training_args.text_pinyin_per_device_train_batch_size
        self.max_len = self.data_args.text_pinyin_block_size
        self.concat_mode = 'segmented'
        self.position_mode = 'aligned'  # aligned or incremental

        self.valid_pinyins = [p.strip('[]') for p in json.load(open(model_args.additional_special_tokens))]
        self.pat = re.compile(r"([\u4e00-\u9fa5]+)")
        self.columns = ['词语']
        if data_args.annotator_tagger == 'pkuseg':
            from pkuseg import pkuseg
            self.tagger = pkuseg()
            # self.tag_to_idx = {v: k for k, v in self.tagger.tagger.idx_to_tag.items()}
        elif data_args.annotator_tagger == 'jieba':
            import jieba
            jieba.dt.tmp_dir = 'data/.cache'
            self.tagger = Tagger(cut=jieba.cut)
        elif data_args.annotator_tagger == 'whitespace':
            self.columns = ['词语']
            self.tagger = Tagger(cut=lambda x: x.split())
        elif data_args.annotator_tagger == 'texsmart':
            import sys
            sys.path.append(f"/apdcephfs/share_916081/minghuantan/p2z/.texsmart-sdk-0.3.5-m-zh/lib")
            from tencent_ai_texsmart import NluEngine

            engine = NluEngine(f'/apdcephfs/share_916081/minghuantan/p2z/.texsmart-sdk-0.3.5-m-zh/data/nlu/kb/', 1)
            options = {"ner": {"enable": False, "fine_grained": False}}

            def cut(text):
                output = engine.parse_text_ext(text, json.dumps(options))
                for item in output.words():
                    # for item in output.phrases():
                    yield item.str

            self.columns = ['词语']
            self.tagger = Tagger(cut=cut)
        elif data_args.annotator_tagger == 'none':
            self.tagger = None
        else:
            assert ValueError('No such tagger given!')

        self.pc_df = get_pinyin_to_char(tokenizer, model_args.pinyin2char_json, model_args.pinyin_logits_processor_cls)

    def parse(self, text, pos, endpos=None):
        endpos = endpos or len(text)
        start = 0
        for item in self.pat.finditer(text, pos=pos, endpos=endpos):
            if item.start() > start and start > pos:
                yield (start, item.start()), text[start:item.start()], False
            start = item.end()
            yield item.span(), item.group(0), True
        if endpos > start:
            yield (start, endpos), text[start:endpos], False

    def parse_with_length(self, context, pivot):
        segments = []
        candidates = collections.defaultdict(list)
        zh_idx_list = []
        min_span_len, max_span_len = 999, 0
        for span, text, is_zh in self.parse(context, pos=pivot):
            span_len = span[1] - span[0]
            if is_zh:
                min_span_len = min(min_span_len, span_len)
                max_span_len = max(max_span_len, span_len)
                candidates[span_len].append(len(segments))
                zh_idx_list.append(len(segments))
            else:
                if span_len > 80 or len(text.split()) > 10:
                    continue
            segments.append((span, text, is_zh))

        return segments, zh_idx_list, candidates

    def build_sample(self, example):
        example = example['json']

        # we also choose abbreviations in other cases
        context = example['content'].replace('\n', '')
        if len(context) < 10:
            raise ValueError(f"Too short context: {len(context)}")

        pivot = random.randint(0, max(0, len(context) - self.max_len))
        segments, zh_idx_list, candidates = self.parse_with_length(context, pivot)
        if len(segments) == 0:
            raise ValueError(f"No segments from context: {context[pivot:]}")

        prob = random.random()
        # we use 1, 2, 3 words as target randomly
        if prob < 0.5:
            target_len = math.ceil(prob / 0.9 * 10)
        else:
            target_len = random.randint(6, 25)

        allowed = list(flatten([v for k, v in candidates.items() if k > target_len]))
        if len(allowed) > 0:
            target_idx = random.choice(allowed)
        else:
            target_len = max(candidates)
            target_idx = random.choice(candidates[target_len])

        pre_context_ids = list(flatten(
            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)) for _, text, _ in
             segments[:target_idx]]))

        _, text, _ = segments[target_idx]
        df_s = pda.DataFrame(self.tagger.cut(text), columns=self.columns)
        df_s = df_s.assign(
            pinyin=df_s['词语'].apply(lambda x: get_pinyin_with_mode(x, self.model_args.abbr_mode)))
        word_boundary = []
        in_context_ids = []
        in_pinyin_ids = []
        for j, item in enumerate(df_s.to_dict('records')):
            offset_start = len(in_context_ids)
            for token, p in zip(item['词语'], item['pinyin']):
                in_context_ids.append(self.tokenizer.convert_tokens_to_ids(token))
                in_pinyin_ids.append(convert_pinyin_to_ids(self.tokenizer, p))
            offset_end = len(in_context_ids)
            word_boundary.append((offset_start, offset_end))

        start_idx = random.choice([s for s, e in word_boundary if len(in_context_ids) - s >= target_len])
        end_idx = [e for s, e in word_boundary if e - start_idx >= target_len][0]

        pre_context_ids = pre_context_ids + in_context_ids[:start_idx]
        post_context_ids = in_context_ids[start_idx:end_idx]
        pinyin_ids = in_pinyin_ids[start_idx:end_idx]

        if len(pinyin_ids) * 2 > self.max_len:
            raise ValueError(f"Pinyin too long: {(start_idx, end_idx)}")

        if len(pre_context_ids) > self.max_len - len(pinyin_ids) * 2:
            pre_context_start = len(pre_context_ids) - (self.max_len - len(pinyin_ids) * 2)
            pre_context_ids = pre_context_ids[pre_context_start:]

        try:
            post_label_ids = [-100 if c == self.tokenizer.unk_token_id else self.pc_df[p].loc[c].idx for p, c in
                              zip(pinyin_ids, post_context_ids)]
            post_gather_ids = [self.pc_df[p].index.to_list() for p in pinyin_ids]
        except KeyError:
            for p, c in zip(pinyin_ids, post_context_ids):
                if c not in self.pc_df[p].index:
                    raise KeyError(f"{p} {c} {self.tokenizer.convert_ids_to_tokens([p, c])}")
            raise KeyError

        assert len(pinyin_ids) == len(post_context_ids)

        if self.concat_mode == 'segmented':
            pre_context_ids = pre_context_ids + [self.tokenizer.sep_token_id]
            pinyin_ids = pinyin_ids + [self.tokenizer.sep_token_id]

        context_position_ids = list(range(1, 1 + len(pre_context_ids)))

        context_ids = pre_context_ids + pinyin_ids + post_context_ids
        label_ids = [-100] * len(pre_context_ids) + [-100] * len(pinyin_ids) + post_label_ids
        gather_ids = [[0]] * (len(pre_context_ids) + len(pinyin_ids)) + post_gather_ids

        pinyin_position_ids = list(range(1 + len(context_position_ids),
                                         1 + len(context_position_ids) + len(pinyin_ids)))
        position_ids = context_position_ids + pinyin_position_ids + pinyin_position_ids

        input_ids = [self.tokenizer.cls_token_id] + context_ids + [self.tokenizer.sep_token_id]
        label_ids = [-100] + label_ids + [-100]
        if self.concat_mode == 'segmented':
            position_ids = [0] + position_ids  # no need to pend
        else:
            position_ids = [0] + position_ids + [len(label_ids) - 1]  # pending one for the final sep token
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids)
        label_ids = torch.tensor(label_ids)
        position_ids = torch.tensor(position_ids)
        attention_mask = torch.tensor(attention_mask)

        if self.position_mode == 'incremental':
            return input_ids, attention_mask, label_ids, gather_ids, max(map(len, gather_ids))
        else:
            return input_ids, attention_mask, label_ids, gather_ids, max(map(len, gather_ids)), position_ids

    def wrap_build_sample(self, example):
        try:
            return self.build_sample(example)
        except Exception as e:
            logger.warning([e, example])
            if self.position_mode == 'incremental':
                return [None] * 5
            else:
                return [None] * 6

    @staticmethod
    def collate_fn(inputs):
        if len(inputs[0]) == 6:
            (input_ids, attention_mask, label_ids, gather_ids, max_gather_len, position_ids) = map(list,
                                                                                                   unzip(
                                                                                                       [item for item in
                                                                                                        inputs if
                                                                                                        None not in item]))
        else:
            (input_ids, attention_mask, label_ids, gather_ids, max_gather_len) = map(list,
                                                                                     unzip(
                                                                                         [item for item in inputs if
                                                                                          None not in item]))
            position_ids = None

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)

        n, l = label_ids.size()

        max_gather_len = max(max_gather_len)

        gather_index = torch.zeros(max_gather_len, dtype=torch.long).unsqueeze(0).repeat(n, l - 1, 1).clone()
        for i, g in enumerate(gather_ids):
            for j, gg in enumerate(g):
                gather_index.data[i, j, :len(gg)] = torch.tensor(gg)

        batch = {'input_ids': input_ids,
                 # 'token_type_ids': token_type_ids,
                 'attention_mask': attn_masks,
                 'gather_index': gather_index,
                 # 'option_ids': torch.tensor(options).long(),
                 'labels': label_ids}

        if position_ids is not None:
            position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
            batch['position_ids'] = position_ids

        return batch

    def __iter__(self):
        assert len(self.shards) >= self.training_args.world_size  # guarantee at least one shard for each device
        logging.info(f"Constructing data loader for text pinyin: {len(self.shards)}")
        dataset = wds.WebDataset(self.shards).shuffle(1000).decode().map(self.wrap_build_sample)
        for batch in DataLoader(dataset, num_workers=8, batch_size=self.batch_size,
                                collate_fn=self.collate_fn):
            yield batch
