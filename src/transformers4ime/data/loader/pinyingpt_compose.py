import math
import random

import msgpack
from lz4.frame import decompress

from transformers4ime.data.loader import register_loader, PinyinGPT2Loader


@register_loader('pinyingpt-compose')
class PinyinGPT2ComposeLoader(PinyinGPT2Loader):

    def context_parsing(self, key, len_list):
        opts = self.config
        len_list = msgpack.loads(decompress(len_list), raw=False) if isinstance(len_list, bytes) else len_list
        if all([sum(len_) == 1 for len_ in len_list]):
            # raise ValueError('Bad Example')
            return (key, (0, 0), (0, 1)), 3

        while True:
            mid = random.randint(0, len(len_list) - 1)
            len_ = len_list[mid]
            if sum(len_) > 1:
                break

        if opts.max_prefix_len > 0:
            pivot = random.randint(0, len(len_) - 1)

            prob = random.random()
            # we use 1, 2, 3 words as target randomly
            if prob < 0.5:
                target_len = math.ceil(prob / 0.9 * 10)
            else:
                target_len = random.randint(6, 25)

            if len(len_) < pivot + target_len:
                target_len = len(len_) - pivot + 1

            while sum(len_[pivot:pivot + target_len]) > opts.max_txt_len:
                target_len -= 1

            pre_context_len = opts.max_txt_len - sum(len_[pivot:pivot + target_len]) - sum(len_[:pivot])
            if pre_context_len < 0:
                return (key, (mid, mid), (pivot, target_len)), opts.max_txt_len + 2

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
            return (key, (start, mid), (pivot, target_len)), opts.max_txt_len + 2 - pre_context_len
        else:
            pivot = 0
            target_len = len(len_)
            while sum(len_[pivot:pivot + target_len]) > opts.max_txt_len:
                target_len -= 1
            return (key, (mid, mid), (pivot, target_len)), sum(len_[pivot:pivot + target_len]) + 2
