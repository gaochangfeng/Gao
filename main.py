import torch
from MyNet.encoder import Encoder

if __name__ == '__main__':
    a = torch.Tensor(10,2,5,3)
    mask = torch.ByteTensor([[0,0,0,0,1],[0,0,0,1,1]])
    encoder = Encoder(3,5,num_blocks=3)
    for i in range(10):
        b,_ = encoder(a[i],mask)
    print(a.size())
    print(b.size())