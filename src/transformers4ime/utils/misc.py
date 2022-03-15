"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc utilities
"""
import json
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from .logger import LOGGER
from ..data.pinyin import get_pinyin_to_char
from ..models import MODEL_REGISTRY
from ..models.cpm2 import CPM2Tokenizer
from ..models.pinyingpt import PinyinGPTLMHeadModel, PinyinGPTConfig


class NoOp(object):
    """ useful for distributed training No-Ops """

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def parse_with_config(parser):
    """
    Parse from config files < command lines < system env
    """
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
            if os.getenv(k.upper()):
                new_v = os.getenv(k.upper())
                if isinstance(v, int):
                    new_v = int(new_v)
                if isinstance(v, float):
                    new_v = float(new_v)
                if isinstance(v, bool):
                    new_v = bool(new_v)
                setattr(args, k, new_v)
                LOGGER.info(f"Replaced {k} from system environment {k.upper()}: {new_v}.")

    # del args.config
    # args.model_config = os.path.join(args.pretrained_model_name_or_path, 'config.json')
    return args


VE_ENT2IDX = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}

VE_IDX2ENT = {
    0: 'contradiction',
    1: 'entailment',
    2: 'neutral'
}


class Struct(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parent_dir(cur_dir):
    return os.path.abspath(os.path.join(cur_dir, os.path.pardir))


def is_word(word):
    for item in list(word):
        if item not in "qwertyuiopasdfghjklzxcvbnm":
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def parse_model_name(model_name, opts):
    assert model_name.startswith('pinyingpt')

    if model_name.startswith('pinyingpt-compose'):
        dataset_cls = benchmark_cls = pinyin_logits_processor_cls = 'pinyingpt-compose'
        if model_name.startswith('pinyingpt-compose-top'):
            model_cls = 'pinyingpt-compose-top'
            if model_name.startswith('pinyingpt-compose-top-residual'):
                opts.compose_mode = 'residual'
            elif model_name.startswith('pinyingpt-compose-top-states'):
                opts.compose_mode = 'states'
            elif model_name.startswith('pinyingpt-compose-top-logits'):
                opts.compose_mode = 'logits'
            else:
                raise ValueError(f"No training model is given, supported models are: {MODEL_REGISTRY.keys()}!")
        elif model_name.startswith('pinyingpt-compose-stack'):
            model_cls = 'pinyingpt-compose-stack'
            opts.num_stack_layers = int(model_name.split('-')[4])
        elif model_name.startswith('pinyingpt-compose-bottom'):
            model_cls = 'pinyingpt-compose-bottom'
        else:
            raise ValueError(f"No training model is given, supported models are: {MODEL_REGISTRY.keys()}!")
    elif model_name.startswith('pinyingpt-concat'):
        model_cls = 'pinyingpt-concat'
        dataset_cls = benchmark_cls = pinyin_logits_processor_cls = 'pinyingpt-concat'
        opts.position_mode = 'aligned'
        if model_name.startswith('pinyingpt-concat-directly'):
            opts.concat_mode = 'directly'
        elif model_name.startswith('pinyingpt-concat-segmented'):
            opts.concat_mode = 'segmented'
            if model_name.startswith('pinyingpt-concat-segmented-positional'):
                opts.position_mode = 'positional'
        else:
            if model_name == "pinyingpt-concat-pcloss-abbr-only":
                opts.concat_mode = 'segmented'
            else:
                raise ValueError(f"No training model is given, supported models are: {MODEL_REGISTRY.keys()}!")

        opts.prefix_loss = True if 'prefix-loss' in model_name else False
    else:
        raise ValueError(f"No training model is given, supported models are: {MODEL_REGISTRY.keys()}!")

    validator_cls = loader_cls = dataset_cls
    if "pcloss" in model_name:
        opts.loss_mode = 'pcloss'
        dataset_cls = f'{dataset_cls}-pcloss'
    else:
        opts.loss_mode = 'ce'

    if model_name.endswith('abbr-only'):
        mode = 'abbr-only'
    elif model_name.endswith('pinyin-only'):
        mode = 'pinyin-only'
    elif model_name.endswith('pinyin-abbr'):
        mode = 'pinyin-abbr'
    else:
        raise ValueError("No training mode is given!")

    dataset_cls = f'{dataset_cls}-{mode}'
    if model_cls == 'pinyingpt-concat' and opts.gpt2_fixed:
        dataset_cls = f'{dataset_cls}-fixed'
        validator_cls = f'{validator_cls}-fixed'

    eval_dataset_cls = f'{dataset_cls}-eval'

    opts.model_cls = model_cls
    opts.dataset_cls = dataset_cls
    opts.loader_cls = loader_cls
    opts.eval_dataset_cls = eval_dataset_cls
    opts.validator_cls = validator_cls
    opts.benchmark_cls = benchmark_cls
    opts.pinyin_logits_processor_cls = pinyin_logits_processor_cls


def construct_model(model_cls, opts):
    LOGGER.info(opts)

    if not opts.pinyin2char_json:
        opts.pinyin2char_json = os.path.join(opts.pretrained_model_name_or_path, "pinyin2char.json")

    if not opts.additional_special_tokens:
        opts.additional_special_tokens = os.path.join(opts.pretrained_model_name_or_path,
                                                      "additional_special_tokens.json")
    with open(opts.additional_special_tokens) as f:
        additional_special_tokens = json.load(f)

    config = PinyinGPTConfig.from_pretrained(opts.pretrained_model_name_or_path)  # config as in the paper
    LOGGER.info(config)
    LOGGER.info(config.hidden_size)

    opts.pinyin_vocab_size = len(additional_special_tokens) + 1
    if model_cls == 'pinyin-cpm-compatible':
        tokenizer = CPM2Tokenizer.from_pretrained(opts.pretrained_model_name_or_path,
                                                  additional_special_tokens=additional_special_tokens)
        ModelCls = MODEL_REGISTRY[model_cls]
        # Prepare model
        model = ModelCls.from_pretrained(opts.pretrained_model_name_or_path, opts=opts)
    else:
        tokenizer = AutoTokenizer.from_pretrained(opts.pretrained_model_name_or_path,
                                                  additional_special_tokens=additional_special_tokens)
        if model_cls == 'gpt2':
            # Prepare model
            try:
                model = PinyinGPTLMHeadModel.from_pretrained(opts.pretrained_model_name_or_path)
            except OSError:
                config = PinyinGPTConfig.from_pretrained(opts.pretrained_model_name_or_path)  # config as in the paper
                model = PinyinGPTLMHeadModel(config, opts=opts)
        elif model_cls == 'pinyingpt-concat':
            model = MODEL_REGISTRY['pinyingpt-concat'].from_pretrained(opts.pretrained_model_name_or_path)
            model.resize_token_embeddings(len(tokenizer))
        else:
            ModelCls = MODEL_REGISTRY[model_cls]
            # Prepare model
            model = ModelCls.from_pretrained(opts.pretrained_model_name_or_path)

    if opts.gpt2_fixed:
        for param in model.transformer.parameters():
            param.requires_grad = False

    opts.additional_special_tokens = additional_special_tokens
    opts.sep_token_id = tokenizer.sep_token_id
    opts.pad_token_id = tokenizer.pad_token_id
    opts.pc_df = get_pinyin_to_char(tokenizer, opts.pinyin2char_json, opts.pinyin_logits_processor_cls)
    opts.pinyin_start_id = tokenizer.convert_tokens_to_ids(additional_special_tokens[0])
    LOGGER.info(model)
    return model, tokenizer
