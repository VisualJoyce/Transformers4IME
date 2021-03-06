import torch
from torch.nn.utils.rnn import pad_sequence

from transformers4ime.data.logits_processor import ConstrainedLogitsProcessor


class PinyinGPTConcatLogitsProcessor(ConstrainedLogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, constraint_id) -> torch.FloatTensor:
        bs, il = input_ids.size()
        # gather index
        gather_index_list = []
        for _ in range(bs):
            pid = constraint_id
            c_df = self.pc_df.get(pid.item())
            if c_df is not None:
                gather_index_list.append(torch.tensor(c_df.index.tolist()))
            else:
                gather_index_list.append(torch.zeros(1))
        gather_index = pad_sequence(gather_index_list, batch_first=True, padding_value=0).long().to(scores.device)

        score = torch.gather(scores, 1, gather_index)

        scores.fill_(-float("inf"))
        scores.scatter_(1, gather_index, score)
        return scores
