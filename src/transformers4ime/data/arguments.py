from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class MMModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    code_model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    best_pt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    additional_special_tokens: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    pinyin2char_json: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )

    pinyin_logits_processor_cls: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )


@dataclass
class MMDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_text_only_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_text_pinyin_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )
    eval_text_pinyin_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "for eval text down stream"
        },
    )
    text_pinyin_task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
        },
    )

    train_code_text_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    eval_code_text_files: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    train_code_only_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_image_only_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    eval_image_text_files: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    train_image_text_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_image_text_cls_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_image_text_matching_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_audio_only_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    train_audio_text_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    eval_audio_text_files: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    train_video_text_files: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    eval_video_text_files: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    mim_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    mfm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked frame(video) modeling loss"}
    )

    text_only_block_size: int = field(
        default=512,
        metadata={
        },
    )

    text_pinyin_block_size: int = field(
        default=256,
        metadata={
        },
    )

    code_text_block_size: int = field(
        default=512,
        metadata={
        },
    )
    code_only_block_size: int = field(
        default=128,
        metadata={
        },
    )
    code_zh_max_len: int = field(
        default=64,
        metadata={
        },
    )

    image_text_block_size: int = field(
        default=128,
        metadata={
        },
    )

    audio_only_sample_rate: int = field(
        default=16000,
        metadata={
        },
    )

    audio_max_duration_in_seconds: int = field(
        default=10,
        metadata={
        },
    )

    audio_min_duration_in_seconds: int = field(
        default=0,
        metadata={
        },
    )

    audio_max_gumbel_temperature: float = field(
        default=2.0,
        metadata={"help": "Maximum temperature for gumbel softmax."},
    )
    audio_min_gumbel_temperature: float = field(
        default=0.5,
        metadata={"help": "Minimum temperature for gumbel softmax."},
    )
    audio_gumbel_temperature_decay: float = field(
        default=0.9995,
        metadata={"help": "Decay of gumbel temperature during training."},
    )

    audio_eval_metrics: Optional[List[str]] = list_field(
        default=['cer'],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )

    audio_tokenizer_ctc: Optional[str] = field(
        default='/apdcephfs/share_916081/duyu_shared_data/pretrained_models/wav2vec2-large-xlsr-53-chinese-zh-cn',
        metadata={
            "help": "The input training data files (multiple files in glob format). "
                    "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
        },
    )

    video_text_block_size: int = field(
        default=128,
        metadata={
        },
    )

    annotator_tagger: Optional[str] = field(
        default="pkuseg",
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )


@dataclass
class MMTrainingArguments(TrainingArguments):
    text_only_per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    text_only_per_device_eval_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    text_only_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    text_pinyin_per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    text_pinyin_per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    text_pinyin_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    image_only_per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    image_only_per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    image_only_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    image_text_per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    image_text_per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    image_text_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    image_text_matching_per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    image_text_cls_per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    audio_only_per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    audio_only_per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    audio_only_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    # for audio
    audio_freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    audio_finetune: Optional[bool] = field(
        default=False, metadata={"help": "Whether to audio finetune."}
    )
    # end for audio

    audio_text_per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    audio_text_per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    audio_text_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    code_text_per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    code_text_per_device_eval_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    code_text_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    code_only_per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    code_only_per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    code_only_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    video_text_per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    video_text_per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    video_text_gradient_accumulation_steps: int = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    use_at_most_k: int = field(
        default=None, metadata={"help": "use at most k training examples for each modality"}
    )

    roll_modalities: bool = field(
        default=False, metadata={"help": "use at most k training examples for each modality"}
    )

    interchange_mode: str = field(
        default='modality_wise', metadata={"help": "use at most k training examples for each modality"}
    )
