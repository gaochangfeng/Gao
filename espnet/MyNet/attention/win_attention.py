import math

import numpy
import torch
from torch import nn
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class WinMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, max_len=5000):
        super(WinMultiHeadedAttention, self).__init__(n_head, n_feat, dropout_rate)
        # assert n_feat % n_head == 0
        # # We assume d_v always equals d_k
        # self.d_k = n_feat // n_head
        # self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)
        # self.linear_out = nn.Linear(n_feat, n_feat)
        # self.attn = None
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        self.win = self.genwindows(self.max_len)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        scale = self.getwintensor(query.size(1), key.size(1)).to(scores.device)
        scores = scores * scale
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, MIN_VALUE)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def genwindows(self, max_len=5000):
        if max_len == 0:
            return None
        x = torch.arange(0, max_len / 2)
        rx = max_len / 2 - torch.arange(0, max_len / 2)
        win = torch.exp(-x * x / (2.0 * (max_len / 4) * (max_len / 4)))
        r_win = torch.exp(-rx * rx / (2.0 * (max_len / 4) * (max_len / 4)))
        return win, r_win

    def getwindowsline(self, len, reverse=False):
        if self.win is None:
            return None
        div = (self.max_len - 1) / (2 * len + 1)
        if reverse:
            return self.win[1][1::int(div)][0:len]
        else:
            return self.win[0][1::int(div)][0:len]

    def getwintensor(self, qlen, klen, qoff=0):
        line_list = []
        if self.win is None:
            win_tensor = torch.ones(qlen, klen)
        else:
            for i in range(qlen):
                l_len = klen - qlen + qoff + i
                r_len = klen - l_len - 1
                l_win = self.getwindowsline(l_len, True)
                r_win = self.getwindowsline(r_len, False)
                line_win = torch.cat([l_win, torch.ones(1), r_win], dim=0)
                line_list.append(line_win)
            win_tensor = torch.stack(line_list).detach()
        win_tensor.requires_grad = False
        return win_tensor


class SmoothMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(SmoothMultiHeadedAttention, self).__init__(n_head, n_feat, dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, MIN_VALUE)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
            if self.training:
                self.attn = self.score_smooth(self.attn).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
            if self.training:
                self.attn = self.score_smooth(self.attn)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def score_smooth(self, scores, eta=0.1):
        time2 = scores.size(-1)
        smooth = torch.ones_like(scores).to(scores.device) / time2
        scores = (1 - eta) * scores + eta * smooth
        return scores
