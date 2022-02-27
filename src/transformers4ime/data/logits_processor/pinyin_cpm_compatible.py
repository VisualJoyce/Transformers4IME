import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import LogitsProcessor

from transformers4ime.data.logits_processor import register_logits_processor


@register_logits_processor("pinyin-cpm-compatible")
class PinyinCPMCompatibleLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, sep_token_id, pc_df, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.sep_token_id = sep_token_id
        self.pad_token_id = kwargs['pad_token_id']
        self.df_vocab, self.pc_df, self.tokenizer = pc_df

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _, l = input_ids.size()
        a = (input_ids == self.sep_token_id).long()

        # LOGGER.debug(a.cumsum(dim=1))

        # context length
        context_length = (a.cumsum(dim=1) < 1).sum(dim=1)

        # pinyin length
        pinyin_length = (a.cumsum(dim=1) == 1).sum(dim=1)
        # pinyin_length = (a.cumsum(dim=1) > 0).sum(dim=1) - (a.cumsum(dim=1) > 1).sum(dim=1)

        # target length
        target_length = (a.cumsum(dim=1) > 1).sum(dim=1)

        # context_ids = [inp[:c] for inp, c in zip(input_ids, context_length)]
        pinyin_ids = [inp[c:c + p] for inp, c, p in zip(input_ids, context_length, pinyin_length)]
        target_ids = [inp[c + p:c + p + t] for inp, c, p, t in
                      zip(input_ids, context_length, pinyin_length, target_length)]

        # LOGGER.debug(input_ids)
        # LOGGER.debug(context_ids)
        # LOGGER.debug(pinyin_ids)
        # LOGGER.debug(target_ids)
        gather_index_list = []
        for p_inp, t_inp in zip(pinyin_ids, target_ids):
            tmp_len = sum([self.df_vocab[self.df_vocab.index == tid.item()].char_len.item() for tid in t_inp[1:]]) + 1
            c_df_index = []
            if tmp_len < len(p_inp):
                # LOGGER.debug(self.tokenizer.decode(p_inp))
                # LOGGER.debug(self.tokenizer.decode(t_inp))
                p_start = p_inp[tmp_len].item()
                # LOGGER.debug((p_start, t_inp))
                c_df = self.pc_df.get(p_start)
                # LOGGER.debug(c_df)
                if c_df is not None:
                    for item in c_df.itertuples():
                        check = True
                        for i, pid in enumerate(item.pinyin_ids):
                            if tmp_len + i < len(p_inp):
                                if pid != p_inp[tmp_len + i].item():
                                    check = False
                                    break
                        if check:
                            c_df_index.append(item.Index)
                    # LOGGER.debug(c_df[c_df.index.isin(c_df_index)])

            if c_df_index:
                # gather_index_list.append(torch.tensor(c_df.index.tolist()))
                gather_index_list.append(torch.tensor(c_df_index))
            else:
                gather_index_list.append(torch.tensor([self.pad_token_id]))
        gather_index = pad_sequence(gather_index_list, batch_first=True, padding_value=self.pad_token_id).long().to(
            scores.device)

        score = torch.gather(scores, 1, gather_index)

        scores.fill_(-float("inf"))
        scores.scatter_(1, gather_index, score)
        return scores
