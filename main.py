import torch
from espnet.MyNet.modules.subencoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

if __name__ == '__main__':
    a = torch.randn(10, 2, 1000, 40)
    srcmask = ~make_pad_mask([1000, 90]).unsqueeze(-2)
    mask = torch.ByteTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).unsqueeze(-2)
    srcmask = ~make_pad_mask([1000, 90]).unsqueeze(-2)
    # encoder = Encoder(40,time_len=10,num_blocks=3,attention_type="traditional")
    encoder = Encoder(40, center_len=64, left_len=64, hop_len=64, right_len=64, num_blocks=6,att_type="rel",
                      rel_pos=0, abs_pos=1, use_mem=1,input_layer="conv2d",subpos=[0,1])
    for i in range(1):
        b, masks = encoder(a[i], srcmask)
    ar = torch.randn(3, 900, 40)
    srcmask2 = ~make_pad_mask([900, 900, 800]).unsqueeze(-2)
    b, masks = encoder(ar, srcmask2)
    print(a.size())
    print(masks.size())

    # for m in encoder.modules():
    #     print(m)

    # sub = NormalSubsamplingPos()
    # sub2 = PoolSubsampling()
    # cx,cm= sub2(a[0],srcmask)
    # print(cx.size(),cm.size())