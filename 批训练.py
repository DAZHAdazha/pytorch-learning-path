import torch
import torch.utils.data as Data

BATCH_SIZE = 5 # 批数据数量

x = torch.linspace(1,10,10) # 10个数据
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y) # x为训练的值, y为要计算误差的目标
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 是否随机打乱
)

# 一共训练3次
for epoch in range(3):
    # 单次size为5 所以把10个数据拆成2份
    for step,(batch_x,batch_y) in enumerate(loader): # 
        # training...
         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())


