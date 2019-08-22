import torch
from MyNet.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

if __name__ == '__main__':
    a = torch.randn(10,2,10,40)
    mask = torch.ByteTensor([[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1]]).unsqueeze(-2)
    encoder = Encoder(40,num_blocks=3,attention_type="traditional")
    #encoder = Encoder(40, num_blocks=3, attention_type="memory")
    for i in range(10):
        b,_ = encoder(a[i],~mask)
    print(a.size())
    print(b)