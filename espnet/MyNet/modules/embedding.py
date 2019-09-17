import math

import torch
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class PositionalEncodingPos(PositionalEncoding):
    """Positional encoding module

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model, dropout_rate, max_len=max_len)

    def forward(self, x, pos=0):
        x = x * self.xscale + self.pe[:, pos:pos + x.size(1)]
        return self.dropout(x)
