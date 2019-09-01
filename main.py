import torch
from espnet.MyNet.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

# from espnet.MyNet.e2e_asr_transformerXL import E2E

if __name__ == '__main__':
    a = torch.randn(10, 2, 1000, 40)

    mask = torch.ByteTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).unsqueeze(-2)
    srcmask = ~make_pad_mask([1000, 90]).unsqueeze(-2)
    # encoder = Encoder(40,time_len=10,num_blocks=3,attention_type="traditional")
    encoder = Encoder(40, time_len=64,mem_len=0,ext_len=64,future_len=64, num_blocks=2,rel_pos=False,abs_pos=True)
    for i in range(3):
        b, masks = encoder(a[i], srcmask)
    ar = torch.randn(3, 600, 40)
    srcmask2 = ~make_pad_mask([600, 90, 800]).unsqueeze(-2)
    b, masks = encoder(ar, srcmask2)
    print(a.size())
    print(b.size())
    print(masks.size())
