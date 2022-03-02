import torch
from transformers import LogitsProcessor


class ConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, sep_token_id, pc_df, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sep_token_id = sep_token_id
        self.pc_df = pc_df

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, constraint_id) -> torch.FloatTensor:
        return scores
