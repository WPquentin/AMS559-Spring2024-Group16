# 1 用带bc的fno和不带bc的fno分别在bc为0.01,0.02和0.04的训练集上训练，获得6组参数，存下来
# 1.1 生成001, 002和004的数据集
# 1.2 写一个程序，参数为001或002或004，它自动读取数据并且训练带bc的fno，并把模型参数保存下来为diffution001_bc.pt，写一个脚本跑三遍
# 1.3 写一个程序，参数为001或002或004，它自动读取数据并且训练不带bc的fno，并把模型参数保存下来为diffution001.pt，写一个脚本跑三遍
# 2 生成B个初始条件，用六个模型各预测一步，然后看它们之间的关系是不是符合理论结果即(a-b)/(b-c)看是不是一个常数向量
# 2.1 生成一个 5*60的向量，每一行是高斯过程的一个sample
# 2.2 把六个模型的参数load一下，并且把5* 60的向量concat成5*120，最后一维是对应的0.01/0.02/0.04
# 2.3 用六个模型跑一下，获得结果 out_bc_001, out_bc_002, out_bc_004; out_001, out_002, out_004
# 2.4 