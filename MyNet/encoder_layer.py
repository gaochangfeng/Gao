import torch
import torch.nn as nn
from MyNet.repos_attention import PositionalEmbedding
from MyNet.repos_attention import RelPartialLearnableDecoderLayer

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

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, ext_len, mem_len, dropatt, pre_lnorm,tgt_len=None):
        super(EncoderLayer, self).__init__()
        self.mems = None
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.mem_len = mem_len
        self.tgt_len = tgt_len
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.rand(size=[n_head, d_head]))
        self.r_r_bias = nn.Parameter(torch.rand(size=[n_head, d_head]))
        self.layer = RelPartialLearnableDecoderLayer(n_head=n_head, d_model=d_model, d_inner=d_inner, d_head=d_head,
                                                     dropout=dropout,
                                                     tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                                     dropatt=dropatt, pre_lnorm=pre_lnorm)
        self.drop = nn.Dropout(dropout)
        self.ext_len= ext_len
    def init_mems(self):
        param = next(self.parameters())
        if self.mem_len > 0:
            self.mems = torch.empty(0, dtype=param.dtype, device=param.device)
        else:
            self.mems = None
        return None

    def _forward(self, dec_inp, mask,mems=None):
        qlen = dec_inp.size(0)
        word_emb = dec_inp
        mlen = mems.size(0) if mems is not None else 0
        klen = mlen + qlen
        dec_attn_mask = mask
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids = core_out

        mems_i = None if mems is None else mems
        core_out = self.layer(core_out, pos_emb, self.r_w_bias,
                              self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        return core_out, new_mems

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None
        # mems is not None
        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            cat = torch.cat([mems, hids], dim=0)
            new_mems = cat[beg_idx:end_idx].detach()
            return new_mems

    def forward(self, x, masks):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.mems is None:
            self.init_mems()
        if self.tgt_len is None:
            tgt_len = x.size(1)
        else:
            tgt_len = self.tgt_len
        x = x.transpose(0, 1)
        hidden, self.mems = self._forward(x,mask=masks,mems=self.mems)
        pred_hid = hidden[-tgt_len:]
        x = pred_hid.transpose(0, 1)
        return x, masks
