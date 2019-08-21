# Gao
ESPnet的Encoder替换为TransformerXL的方式
保留了原ESPnet Encoder的大多数使用接口，但定义Encoder时需要指定输出序列的长度
保留了ESPnet的mask的功能，当输入数据没有被mask时，可以输入None加快速度

# 使用方法：
1.找到需要将传统Encoder替换为TransformerXL的地方
2.import的部分
    #from espnet.nets.pytorch_backend.transformer.encoder import Encoder
    from MyNet.encoder import Encoder
此时，已经变成了记忆长度为0的相对位置编码下的Encoder
3.找到Encoder定义的位置，设定Encoder的记忆长度
    encoder = Encoder(3,time_len=5,ext_len=3,num_blocks=3)
    
# 问题
 1.部分冗余的构造函数没有删除
 2.由于mems信息保存在每一层的的数据成员中，程序不能使用DataParallel进行并行计算
