"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import csv
import glob
import json
import logging
import random
from dataclasses import dataclass
from typing import Optional

from transformers import PretrainedConfig

from transformers4ime.data.arguments import MMModelArguments, MMTrainingArguments, MMDataTrainingArguments

# from typing import Tuple, Any, OrderedDict, Optional, List, Union

logger = logging.getLogger(__name__)


class IMEBaseDataLoader(object):

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

    @staticmethod
    def get_shards(train_files):
        shards = []
        [shards.extend(glob.glob(f"{f}/**/*.tar", recursive=True)) for f in train_files.split(":")]
        return sorted(shards)  # 所有tar文件路径

    def do_mask_and_convert(self, sent: list, tokenizer):
        # sent list of strings (tokens)
        # make masks & convert to word-ids

        # TODO: make word-inputs to char ids
        # TODO:
        masked_sent = []
        truth = []

        # tokens = tokenizer.tokenize(sent)  # learned tokens
        tokens = sent
        # char-to-char; word-2-word (no extending)
        for token in tokens:
            if random.random() < self.data_args.mlm_probability:
                # do mask
                masked_sent.append(tokenizer._convert_token_to_id(tokenizer.mask_token))
                truth.append(tokenizer._convert_token_to_id(token))
            else:
                masked_sent.append(tokenizer._convert_token_to_id(token))
                truth.append(-100)
        return masked_sent, truth


# for text down stream
@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    # def to_json_string(self):
    #     """Serializes this instance to a JSON string."""
    #     return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, process_index):
        self.process_index = process_index

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines
