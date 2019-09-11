import torch
import torch.nn as nn
from espnet.MyNet.repos_attention import CashEncoderLayer
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
#from espnet.MyNet.attention.win_attention import  WinMultiHeadedAttention as MultiHeadedAttention

class EncoderLayer(nn.Module):
    """Encoder layer module


    :param int size: input dim
    :param int n_head: the number of the attention head
    :param int d_model: the dim of the model
    :param int d_head:  the dim of the head, make sure that d_model == n_head*d_head
    :param int d_inner: the dim of the input data
    :param int tgt_len: the length of the output, None mean the output length is equal to the input
    :param int ext_len: the extra length of memory
    :param int mem_len: the length of the memory,the real memory length is ext_len+mem_len
    :param int dropatt: the rate to drop attention
    :param int pre_lnorm: the way to normalise the data
    """

    def __init__(self, n_head, d_model, d_head, pos_ff,
                 dropout, dropatt, pre_lnorm, tgt_len=None,
                 ext_len=0, mem_len=0, future_len=0, rel_pos=True):
        super(EncoderLayer, self).__init__()
        self.register_buffer('mems', None)
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.mem_len = mem_len
        self.rel_pos = rel_pos
        self.future_len = future_len
        self.tgt_len = tgt_len

        self.layer = CashEncoderLayer(d_model, MultiHeadedAttention(n_head, d_model, dropatt),
                                      pos_ff, dropout, pre_lnorm, concat_after=False)

        self.drop = nn.Dropout(dropout)
        self.ext_len = ext_len

    def init_mems(self):
        param = next(self.parameters())
        if self.mem_len > 0:
            self.mems = torch.empty(0, dtype=param.dtype, device=param.device)
        else:
            self.mems = None

    def _update_mems(self, hids, mems):
        # does not deal with None
        if mems is None: return None
        # mems is not None
        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            if mems.dim() > 1:
                mem_len = mems.size(1)
            else:
                mem_len = 0
            end_idx = mem_len + max(0, self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            cat = torch.cat([mems, hids], dim=1)
            new_mems = cat[:, beg_idx:end_idx].detach().to(hids.device)
            return new_mems

    def forward(self, x, masks):
        """Compute encoded features

        :param masks:
        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, 1, max_time_in),1 for padding,0 for data
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.mems is None and self.mem_len > 0:
            self.init_mems()
        hidden, _ = self.abs_forward(x, masks)
        tgt_len = x.size(1)
        x = hidden[:, -tgt_len:]
        return x, masks

    def abs_forward(self, x, masks):
        x, masks = self.layer(x, self.mems, masks)
        self.mems = self._update_mems(x, self.mems)
        return x, masks
