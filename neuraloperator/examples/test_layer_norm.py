import torch.nn as nn
import torch
import pdb
def bp():
    import pdb; pdb.set_trace()

bp()
# 创建 LayerNorm 层
layer_norm = nn.LayerNorm(normalized_shape=10)

# 假设输入是 10 维向量
x = torch.randn(3, 5, 10)

# 前向传播计算
output = layer_norm(x)

# 获取均值和方差
mean = layer_norm.running_mean
var = layer_norm.running_var