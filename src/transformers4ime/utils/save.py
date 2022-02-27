"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import glob
import json
import os
import subprocess
from collections import Counter
from os.path import abspath, dirname, join

import torch

from .logger import LOGGER


def save_training_meta(args):
    if args.rank > 0:
        return

    os.makedirs(join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(join(args.output_dir, 'ckpt'), exist_ok=True)

    with open(join(args.output_dir, 'log', 'hps.json'), 'w') as writer:
        hps = {k: v for k, v in vars(args).items() if k not in ['tokenizer', 'pc_df']}
        json.dump(hps, writer, indent=4)

    # if os.path.isfile(args.model_config):
    #     model_config = json.load(open(args.model_config))
    #     with open(join(args.output_dir, 'log', 'model.json'), 'w') as writer:
    #         json.dump(model_config, writer, indent=4)
    # git info
    try:
        LOGGER.info("Waiting on git info....")
        c = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_branch_name = c.stdout.decode().strip()
        LOGGER.info("Git branch: %s", git_branch_name)
        c = subprocess.run(["git", "rev-parse", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_sha = c.stdout.decode().strip()
        LOGGER.info("Git SHA: %s", git_sha)
        git_dir = abspath(dirname(__file__))
        git_status = subprocess.check_output(
            ['git', 'status', '--short'],
            cwd=git_dir, universal_newlines=True).strip()
        with open(join(args.output_dir, 'log', 'git_info.json'),
                  'w') as writer:
            json.dump({'branch': git_branch_name,
                       'is_dirty': bool(git_status),
                       'status': git_status,
                       'sha': git_sha},
                      writer, indent=4)
    except subprocess.TimeoutExpired as e:
        LOGGER.exception(e)
        LOGGER.warn("Git info not found. Moving right along...")
    except subprocess.CalledProcessError as e:
        LOGGER.exception(e)
        LOGGER.warn("Git info not found. Moving right along...")


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt', keep_only=None):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix
        self.keep_only = keep_only
        self.nbest_counter = Counter()

    def save(self, model, val_loss, step, optimizer=None):
        output_model_file = join(self.output_dir, f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        if hasattr(model, 'vocab_pad') and model.vocab_pad:
            # store vocab embeddings before padding
            emb_w = state_dict['bert.embeddings.word_embeddings.weight']
            emb_w = emb_w[:-model.vocab_pad, :]
            state_dict['bert.embeddings.word_embeddings.weight'] = emb_w
            state_dict['cls.predictions.decoder.weight'] = emb_w

        torch.save(state_dict, output_model_file)
        self.nbest_counter.update({output_model_file: val_loss})

        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')

        models_all = glob.glob(join(self.output_dir, f"{self.prefix}_*.{self.suffix}"))
        if len(models_all) > self.keep_only:
            for f, _ in self.nbest_counter.most_common(len(models_all) - self.keep_only):
                LOGGER.info(f"Removing model file: {f} !")
                os.remove(f)
                self.nbest_counter.pop(f)
