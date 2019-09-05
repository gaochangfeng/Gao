import torch
import torch.nn as nn
from espnet.MyNet.repos_attention import RelPositionalEmbedding
from espnet.MyNet.repos_attention import RelPartialLearnableDecoderLayer
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer as AbsEncoderLayer
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


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
        self.register_buffer('mems',None)
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.mem_len = mem_len
        self.rel_pos = rel_pos
        self.future_len = future_len
        self.tgt_len = tgt_len
        if rel_pos:
            self.layer = RelPartialLearnableDecoderLayer(n_head=n_head, d_model=d_model, d_head=d_head,
                                                         dropout=dropout, pos_ff=pos_ff,
                                                         tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                                         dropatt=dropatt, pre_lnorm=pre_lnorm)
            self.pos_emb = RelPositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.rand(size=[n_head, d_head]))
            self.r_r_bias = nn.Parameter(torch.rand(size=[n_head, d_head]))
        else:
            self.layer = AbsEncoderLayer(d_model, MultiHeadedAttention(n_head, d_model, dropatt),
                                         pos_ff, dropout, pre_lnorm, concat_after=False)
            self.pos_emb = None
            self.r_w_bias = None
            self.r_r_bias = None
        self.drop = nn.Dropout(dropout)
        self.ext_len = ext_len

    def init_mems(self):
        param = next(self.parameters())
        if self.mem_len > 0:
            self.mems = torch.empty(0, dtype=param.dtype, device=param.device)
        else:
            self.mems = None

    def _forward(self, dec_inp, mask, mems=None):
        #print(dec_inp.size(),mask.size())
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
        mems_i = None if mems is None else mems
        core_out = self.layer(core_out, pos_emb, self.r_w_bias,
                              self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        core_out = self.drop(core_out)
        #print('rel',core_out.size())
        return core_out

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
            end_idx = self.mem_len + max(0, self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            cat = hids
            new_mems = cat[beg_idx:end_idx].detach().to(hids.device)
            # if self.rel_pos:
            #     new_mems = cat[beg_idx:end_idx].detach().to(hids.device)
            # else:
            #     new_mems = cat[:,beg_idx:end_idx].detach().to(hids.device)
            return new_mems

    def forward(self, x, masks):
        """Compute encoded features

        :param masks:
        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, 1, max_time_in),1 for padding,0 for data
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        #print("in", x.size(), masks.size())
        if self.mems is None and self.mem_len>0:
            self.init_mems()
        if self.rel_pos:
            hidden,_ = self.rel_forward(x,masks)
        else:
            hidden,_ = self.abs_forward(x,masks)
        tgt_len = x.size(1)
        x = hidden[:,-tgt_len:]
        #print("out", x.size(), masks.size())
        return x, masks

    def rel_forward(self, x, masks):
        if masks is not None:
            in_mask = ~masks.squeeze(1)
        else:
            in_mask = None
        x = x.transpose(0, 1)
        hidden = self._forward(x, mask=in_mask, mems=self.mems)
        qlen = x.size(0)
        self.mems = self._update_mems(x, self.mems, qlen, self.mem_len)
        x = hidden.transpose(0, 1)
        return x,masks

    def abs_forward(self, x, masks):
        qlen = x.size(1)
        if self.mems is not None and self.mems.dim() > 1:
            x = torch.cat([self.mems.transpose(0, 1), x], dim=1)
            if self.mem_len > 0:
                m_mask = torch.ones(masks.size(0), 1, self.mems.size(0)).byte().to(masks.device)
                masks = torch.cat([m_mask, masks], dim=-1)
            x, masks = self.layer(x, masks)
            self.mems = self._update_mems(x.transpose(0,1), self.mems, qlen, self.mem_len)
        elif self.mems is not None:
            x, masks = self.layer(x, masks)
            self.mems = self._update_mems(x.transpose(0, 1), self.mems, qlen, self.mem_len)
        else:
            x, masks = self.layer(x, masks)
        return x,masks
