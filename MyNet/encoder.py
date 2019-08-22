import torch
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer as EncoderLayerTD
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from MyNet.encoder_layer import EncoderLayer as EncoderLayerXL


class Encoder(torch.nn.Module):
    """TransformerXL encoder module,@ref "Transformer-XL_AttentiveLanguageModels BeyondaFixed-LengthContext"
    :param int idim: input dim
    :param int time_len: the mems length of the recurrent structure
    :param int ext_len: the ext_mems length of the recurrent structure
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward FixMe not be used
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding FixMe not be used
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output FixMe not be used
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, idim, time_len=0, ext_len=0,
                 attention_type="memory",
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
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.dropout_rate = dropout_rate
        self.input_layer = input_layer
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.attention_type = attention_type
        self.positional_dropout_rate = positional_dropout_rate
        self.pos_enc_class = pos_enc_class
        self._generateInputLayer()
        if attention_type == "memory":
            self.encoders = repeat(
                num_blocks,
                lambda: EncoderLayerXL(
                    n_head=attention_heads,
                    d_model=attention_dim,
                    d_head=attention_dim // attention_heads,
                    ext_len=ext_len,
                    mem_len=time_len,
                    dropout=dropout_rate,
                    dropatt=attention_dropout_rate,
                    pre_lnorm=normalize_before,
                    pos_ff=PositionwiseFeedForward(attention_dim, linear_units, dropout_rate)
                )
            )
        elif attention_type == "traditional":
            self.encoders = repeat(
                num_blocks,
                lambda: EncoderLayerTD(
                    attention_dim,
                    MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after
                )
            )
        else:
            ValueError("only memory or traditional can be used")
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks=None):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: (batch,1,time),1 means data,zero means padding
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
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
            if self.attention_type == "traditional":
                self.embed.add_module(name="pos_enc", module=self.pos_enc_class(self.attention_dim, self.positional_dropout_rate))
        elif self.input_layer == "conv2d":
            self.embed = Conv2dSubsampling(self.idim, self.attention_dim, self.dropout_rate)
        elif self.input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(self.idim, self.attention_dim),
            )
            if self.attention_type == "traditional":
                self.embed.add_module("pos_enc", self.pos_enc_class(self.attention_dim, self.positional_dropout_rate))
        elif isinstance(self.input_layer, torch.nn.Module):
            self.embed = self.input_layer
        else:
            raise ValueError("unknown input_layer: " + self.input_layer)
