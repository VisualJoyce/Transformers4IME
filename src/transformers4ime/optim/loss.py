import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
                  for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NTXentLoss(nn.Module):
    """
    Code from https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class ContrastiveLoss(nn.Module):

    def __init__(self, tau):
        super().__init__()
        self.temperature = tau

    def forward(self, out_1, out_2):
        batch_size = out_1.size(0)
        out_1, out_2 = F.normalize(out_1, dim=-1), F.normalize(out_2, dim=-1)

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return - torch.log(pos_sim / sim_matrix.sum(dim=-1))


#
# class ContrastiveLoss(nn.Module):
# # class NT_Xent(nn.Module):
#     """
#     https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py
#     """
#
#     def __init__(self, tau):
#         super().__init__()
#         self.temperature = tau
#
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)
#
#     def mask_correlated_samples(self, batch_size, world_size):
#         N = 2 * batch_size * world_size
#         mask = torch.ones((N, N), dtype=bool)
#         mask = mask.fill_diagonal_(0)
#         for i in range(batch_size * world_size):
#             mask[i, batch_size + i] = 0
#             mask[batch_size + i, i] = 0
#         return mask
#
#     def forward(self, z_i, z_j):
#         """
#         We do not sample negative examples explicitly.
#         Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
#         """
#         batch_size = z_i.size(0)
#         world_size = 1
#         mask = self.mask_correlated_samples(batch_size, world_size)
#         N = 2 * batch_size * world_size
#
#         z = torch.cat((z_i, z_j), dim=0)
#         # if self.world_size > 1:
#         #     z = torch.cat(GatherLayer.apply(z), dim=0)
#
#         sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
#
#         sim_i_j = torch.diag(sim, batch_size * world_size)
#         sim_j_i = torch.diag(sim, -batch_size * world_size)
#
#         # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
#         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
#             N, 1
#         )
#         negative_samples = sim[mask].reshape(N, -1)
#
#         labels = torch.zeros(N).to(positive_samples.device).long()
#         logits = torch.cat((positive_samples, negative_samples), dim=1)
#         loss = self.criterion(logits, labels)
#         loss /= N
#         return loss
#
#
# class ContrastiveLoss(nn.Module):
#     """
#     https://github.com/mdiephuis/SimCLR/blob/master/loss.py
#     """
#
#     def __init__(self, tau=1, normalize=False):
#         super(ContrastiveLoss, self).__init__()
#         self.tau = tau
#         self.normalize = normalize
#
#     def forward(self, xi, xj):
#
#         x = torch.cat((xi, xj), dim=0)
#
#         is_cuda = x.is_cuda
#         sim_mat = torch.mm(x, x.T)
#         if self.normalize:
#             sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
#             sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
#
#         sim_mat = torch.exp(sim_mat / self.tau)
#
#         # no diag because it's not diffrentiable -> sum - exp(1 / tau)
#         # diag_ind = torch.eye(xi.size(0) * 2).bool()
#         # diag_ind = diag_ind.cuda() if use_cuda else diag_ind
#
#         # sim_mat = sim_mat.masked_fill_(diag_ind, 0)
#
#         # top
#         if self.normalize:
#             sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
#             sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
#         else:
#             sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)
#
#         sim_match = torch.cat((sim_match, sim_match), dim=0)
#
#         norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
#         norm_sum = norm_sum.cuda() if is_cuda else norm_sum
#         loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
#
#         return loss


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2,
                 reduction: str = "none"):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean() if self.reduction == "mean" else loss
