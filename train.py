import torch
from torch import nn, optim
from utils import get_data_loaders, get_model

def main():

    # 获取数据加载器
    train_loader, _ = get_data_loaders()

    # 获取模型和设备
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义损失函数和优化器
    # criterion是损失函数，用来衡量模型预测结果与实际目标之间的差距
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失(适合分类问题)
    # 优化器:根据 损失函数计算出来的误差来调整模型的参数 Adam是常用的优化器 lr:学习效率(越大训练越快 但是太大太小都会降低精度)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
    # 训练轮数:模型利用数据集进行学习的次数    
    num_epochs = 10  
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        # inputs:当前批次中的输入数据 labels:当前批次的标签 从这行开始会真正进行数据的加载和预处理
        for inputs, labels in train_loader:
            # 将数据移到GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 正向传播 
            # 将输入数据inputs 通过模型model生成预测值outputs
            outputs = model(inputs) # "做题"
            # 根据预测值outputs 和真实标签labels，计算损失loss
            loss = criterion(outputs, labels) # "对答案"

            # 反向传播
            # PyTorch 背后有一个全局的 参数存储机制 所以不用传参数optimizer也能取得loss的回测结果
            loss.backward() # "反思解题思路"
            # 优化器根据解题思路调整模型的参数 
            optimizer.step() # "以后遇到类似的题型不要再出错"
            # item()是 PyTorch 的方法，把tensor类型的 loss 转换成数字 例如： tensor(2.5) -> 2.5
            # 计算整个训练过程中的累计损失
            running_loss += loss.item()

            # 计算准确率
            '''
            假如outputs = [[0.8, 0.2],  狗的概率是0.8 猫0.2  
                           [0.3, 0.7]  狗的概率是0.3 猫0.7
                            ...]
                用torch.max找出各类别的最大概率(_) 和 预测结果(predicted) 参数1是按行单位找(适用于分类任务) 0的话按列找
                所以_是0.8,0.7,predicted是0,1
            '''
            _, predicted = torch.max(outputs, 1)
            ''' 
            统计当前批次中的样本数量
            size里通常有四种数据(batch_size, channels, height, width)  batch_size:这批的样本数
            '''
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # len(train_loader):一共几批
        # 平均每个批次的损失
        epoch_loss = running_loss / len(train_loader)
        # 正确率=所有样本里的正确个数/样本总数
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    print('Finished Training')
    # 保存训练好的模型
    torch.save(model.state_dict(), 'cat_dog_model.pth')

if __name__ == "__main__":
    main()





