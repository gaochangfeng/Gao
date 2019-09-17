import torch

from espnet.MyNet.modules.embedding import PositionalEncodingPos


class Conv2dSubsamplingPos(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate):
        super(Conv2dSubsamplingPos, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)
        )
        self.embed = PositionalEncodingPos(odim, dropout_rate)

    def forward(self, x, x_mask, pos=0):
        """Subsample x

        :param pos: start position
        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x = self.embed(x, pos)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class NormalSubsamplingPos(torch.nn.Module):
    """Normal subsampling

    :param int samp_div: input subsample rate to 1/samp_div
    """

    def __init__(self, samp_div = 2):
        super(NormalSubsamplingPos, self).__init__()
        self.samp_div = samp_div

    def forward(self, x, x_mask=None):
        x = x[:, 0::self.samp_div]
        if x_mask is not None:
            x_mask = x_mask[:,:, 0::self.samp_div]
        return x, x_mask[:,:,:x.size(1)]


class PoolSubsampling(torch.nn.Module):
    def __init__(self, samp_div = 2,type = "max"):
        super(PoolSubsampling, self).__init__()
        self.samp_div = samp_div
        if type == "max":
            self.pool = torch.nn.MaxPool1d(samp_div, stride=samp_div)
        elif type == "average":
            self.pool = torch.nn.AvgPool1d(samp_div, stride=samp_div)

    def forward(self, x, x_mask=None):
        x = x.transpose(1,2) # b c t
        x = self.pool(x)
        x = x.transpose(1,2).contiguous()
        if x_mask is not None:
            x_mask = x_mask[:,:, 0::self.samp_div]
        return x, x_mask[:,:,:x.size(1)]