import torch
from MyNet.my_net import Encoder

if __name__ == '__main__':
    a = torch.Tensor(2,5,3)
    mask = torch.ByteTensor([[0,0,1,1,1],[0,1,1,1,1]])
    encoder = Encoder(3,5,num_blocks=1)
    for i in range(3):
        b,_ = encoder(a,mask)
    print(a.size())
    print(b)