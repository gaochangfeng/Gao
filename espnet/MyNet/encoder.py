import torch
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
#from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.MyNet.subsampling import Conv2dSubsamplingPos as Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.MyNet.encoder_layer import EncoderLayer


class Encoder(torch.nn.Module):
    """TransformerXL encoder module,@ref "Transformer-XL_AttentiveLanguageModels BeyondaFixed-LengthContext"
    :param int idim: input dim
    :param int time_len: the mems length of the recurrent structure
    :param int hop_len: the ext_mems length of the recurrent structure
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, idim, center_len=8, left_len=0, hop_len=0, right_len=0,
                 abs_pos=True, rel_pos=False, use_mem=False,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="conv2d",
                 pos_enc_class=PositionalEncoding,
                 normalize_before=True,
                 concat_after=False):
        super(Encoder, self).__init__()
        self.idim = idim
        self.center_len = center_len
        self.use_mem = use_mem !=0
        self.left_len = left_len
        if self.use_mem != 0:
            self.mem_len = left_len
        else:
            self.mem_len = 0
        self.hop_len = hop_len
        self.right_len = right_len
        self.abs_pos = abs_pos != 0
        self.rel_pos = rel_pos != 0
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.dropout_rate = dropout_rate
        self.input_layer = input_layer
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.positional_dropout_rate = positional_dropout_rate
        self.pos_enc_class = pos_enc_class
        self._generateInputLayer()

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                n_head=attention_heads,
                d_model=attention_dim,
                d_head=attention_dim // attention_heads,
                ext_len=hop_len // 4,
                mem_len=self.mem_len // 4,
                tgt_len=center_len,
                future_len=right_len,
                rel_pos=rel_pos,
                dropout=dropout_rate,
                dropatt=attention_dropout_rate,
                pre_lnorm=normalize_before,
                pos_ff=PositionwiseFeedForward(attention_dim, linear_units, dropout_rate)
            )
        )

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def _forward(self, xs, masks=None, pos=0):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor，(batch,time,dim)
        :param torch.Tensor masks: (batch,1,time),1 means data,zero means padding
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks, pos)
        else:
            xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def _generateInputLayer(self):
        if self.input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(self.idim, self.attention_dim),
                torch.nn.LayerNorm(self.attention_dim),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.ReLU(),
            )
            if self.abs_pos:
                self.embed.add_module(name="pos_enc",
                                      module=self.pos_enc_class(self.attention_dim, self.positional_dropout_rate))
        elif self.input_layer == "conv2d":
            self.embed = Conv2dSubsampling(self.idim, self.attention_dim, self.dropout_rate)
        elif self.input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(self.idim, self.attention_dim),
            )
            if self.abs_pos:
                self.embed.add_module("pos_enc", self.pos_enc_class(self.attention_dim, self.positional_dropout_rate))
        elif isinstance(self.input_layer, torch.nn.Module):
            self.embed = self.input_layer
        else:
            raise ValueError("unknown input_layer: " + self.input_layer)

    def chunkdevide(self, xs, masks):
        chunk_left = self.left_len
        if self.use_mem:
            chunk_left=0
        r_xs = torch.ones(xs.size(0), self.right_len, xs.size(2)).to(xs.device)
        r_masks = torch.zeros(masks.size(0), 1, self.right_len).byte().to(masks.device)
        l_xs = torch.ones(xs.size(0), chunk_left, xs.size(2)).to(xs.device)
        l_masks = torch.zeros(masks.size(0), 1, chunk_left).byte().to(masks.device)
        xs = torch.cat([l_xs, xs, r_xs], dim=1)
        masks = torch.cat([l_masks, masks, r_masks], dim=2)
        m_chunk = []
        m_chunk_mask = []
        i = 0
        while (i + chunk_left + self.center_len + self.right_len) < xs.size(1):
            m_chunk.append(xs[:, i:i + chunk_left + self.center_len + self.right_len])
            m_chunk_mask.append(masks[:, :, i:i + chunk_left + self.center_len + self.right_len])
            i = i + self.hop_len
        m_chunk.append(xs[:, i:])
        m_chunk_mask.append(masks[:, :, i:])
        return m_chunk, m_chunk_mask

    def forward(self, xs, masks=None):
        chunk_left = self.left_len
        if self.use_mem:
            chunk_left=0
        if masks is None:
            masks = torch.ones(xs.size(0), 1, xs.size(1)).byte().to(xs.device)
        if self.center_len == 0:
            xs, masks = self._forward(xs, masks)
        else:
            xs_list = []
            mask_list = []
            chunks, chunks_mask = self.chunkdevide(xs, masks)
            for i in range(len(chunks)):
                if self.abs_pos:
                    xss, maskss = self._forward(chunks[i], chunks_mask[i], i*self.hop_len)
                else:
                    xss, maskss = self._forward(chunks[i], chunks_mask[i], 0)
                xss = xss[:, chunk_left // 4:(chunk_left + self.center_len) // 4]
                maskss = maskss[:, :, chunk_left // 4:(chunk_left + self.center_len) // 4]
                xs_list.append(xss)
                mask_list.append(maskss)
            xs = torch.cat(xs_list, dim=1)
            masks = torch.cat(mask_list, dim=2)
            if self.use_mem:
                for layer in self.encoders._modules.values():
                    if isinstance(layer, EncoderLayer):
                        layer.mems = None
        return xs, masks
