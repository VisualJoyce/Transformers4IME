import json

import pandas as pda
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_tone3
from pypinyin.phrases_dict_large import phrases_dict
from pypinyin.pinyin_dict import pinyin_dict


def get_pinyin_vocab():
    pinyin_with_tone_vocab = set()
    pinyin_no_tone_vocab = set()
    for item in pinyin_dict.values():
        for k in item.split(','):
            p = to_tone3(k, neutral_tone_with_5=True)
            pinyin_with_tone_vocab.add(p)
            pinyin_no_tone_vocab.add(p[:-1])
    for item in phrases_dict.values():
        for k in item:
            p = to_tone3(k[0], neutral_tone_with_5=True)
            pinyin_with_tone_vocab.add(p)
            pinyin_no_tone_vocab.add(p[:-1])
    return sorted(list(pinyin_with_tone_vocab)), sorted(list(pinyin_no_tone_vocab))


def get_pinyin(x):
    return lazy_pinyin(x, style=Style.TONE3, neutral_tone_with_five=True)


def get_wadegiles(x):
    return lazy_pinyin(x, style=Style.WADEGILES)


def wrap_pinyin_to_tokens(pl):
    if isinstance(pl, str):
        return f"[{pl.replace('ü', 'v')}]"
    else:
        return [f"[{pi.replace('ü', 'v')}]" for pi in pl]


def convert_pinyin_to_ids(tokenizer, pl):
    return tokenizer.convert_tokens_to_ids(wrap_pinyin_to_tokens(pl))


def get_pinyin_with_mode(words, abbr_mode):
    pinyin = get_pinyin(words)
    if abbr_mode == 'xone':
        pinyin = [p[:-1] if i == 0 else p[0] for i, p in enumerate(pinyin)]
    elif abbr_mode == 'full':
        pinyin = [p[0] for p in pinyin]
    elif abbr_mode == 'none':
        pinyin = [p[:-1] for p in pinyin]  # remove tone
    else:
        raise ValueError("No such abbr mode!")
    return pinyin


def get_pinyin_to_char(tokenizer, opts):
    with open(opts.pinyin2char_json) as f:
        pinyin2char = json.load(f)

    # pinyin2char = {}
    # for w in tokenizer.get_vocab():
    #     p_list = pinyin(w, heteronym=True, style=Style.TONE3, neutral_tone_with_five=True)
    #     if len(p_list) == 1 and p_list[0][0] != w:
    #         for p in p_list[0]:
    #             pinyin2char.setdefault(p[0], set())
    #             pinyin2char[p[0]].add(w)
    #
    #             pinyin2char.setdefault(p[:-1], set())
    #             pinyin2char[p[:-1]].add(w)

    if opts.pinyin_logits_processor_cls == 'pinyin-cpm-compatible':
        df_vocab = pda.DataFrame(tokenizer.convert_ids_to_tokens(range(len(tokenizer.get_vocab()))), columns=['word'])
        df_vocab = df_vocab.assign(
            char=df_vocab.word.apply(lambda x: x[1] if x.startswith('▁') and len(x) > 1 else x[0]),
            pinyin=df_vocab.word.apply(
                lambda x: get_pinyin_with_mode(x[1:] if x.startswith('▁') and len(x) > 1 else x, abbr_mode='none')))
        df_vocab = df_vocab.assign(char_len=df_vocab.pinyin.apply(len),
                                   pinyin_ids=df_vocab.pinyin.apply(lambda x: convert_pinyin_to_ids(tokenizer, x)))

        pc_df = {}
        for p, wl in pinyin2char.items():
            p_id = convert_pinyin_to_ids(tokenizer, p)
            df_tmp = df_vocab[df_vocab.char.isin(wl)]
            df_tmp = df_tmp.assign(idx=range(len(df_tmp)))
            pc_df[p_id] = df_tmp
        return df_vocab, pc_df, tokenizer
    else:
        df_vocab = pda.DataFrame(tokenizer.convert_ids_to_tokens(range(len(tokenizer.get_vocab()))), columns=['char'])

        pc_df = {}
        for p, wl in pinyin2char.items():
            p_id = convert_pinyin_to_ids(tokenizer, p)
            df_tmp = df_vocab[df_vocab.char.isin(wl)]
            df_tmp = df_tmp.assign(idx=range(len(df_tmp)))
            pc_df[p_id] = df_tmp
        return pc_df
