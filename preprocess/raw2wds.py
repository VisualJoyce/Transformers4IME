"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import argparse
import ast
import glob
import re

import webdataset as wds
from tqdm import tqdm

zh_pat = re.compile(r"([\u4e00-\u9fa5]+)")


def parse(text):
    start = 0
    for item in zh_pat.finditer(text):
        if item.start() > start:
            yield (start, item.start()), text[start:item.start()], False
        start = item.end()
        yield item.span(), item.group(0), True
    if len(text) > start:
        yield (start, len(text)), text[start:len(text)], False


def parse_with_length(context):
    for _, text, is_zh in parse(context):
        if not is_zh:
            for t in text.split('\n'):
                if all([len(t) < 20,
                        len(t.split()) <= 10]):
                    yield t
        else:
            yield text


def main(opts):
    with wds.ShardWriter(
            f"/apdcephfs/share_916081/minghuantan/corpus/cleanmix_corpus/%08d.tar",
            maxcount=100000) as sink:
        for i, text_json in tqdm(enumerate(glob.glob(
                "/apdcephfs/share_916081/duyu_shared_data/chinese_text_raw_data/clean_mix/merged_data/train.txt*"))):
            print(text_json)
            for j, sample in enumerate(open(text_json)):
                try:
                    data = ast.literal_eval(sample)
                except SyntaxError:
                    print(sample)
                    continue

                text = data['content']
                new_text = ''.join([t for t in parse_with_length(text)])
                if new_text.strip():
                    data['content'] = new_text
                    sample = {
                        "__key__": f"{i}-{j}",
                        "json": data,
                    }
                    sink.write(sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--annotator_tagger', default="texsmart", help='annotation JSON')
    # parser.add_argument('--genre', required=True, help='annotation JSON')
    # parser.add_argument('--config', help='JSON config files')
    # parser.add_argument('--cores', type=int, default=os.cpu_count(), help='JSON config files')
    args = parser.parse_args()
    main(args)
