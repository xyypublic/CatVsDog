import torch
from torch import nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 数据预处理和加载
def get_data_loaders():
    train_dir = '文件路径'
    test_dir = '文件路径'
    batch_size = 32  # 每次分析多少张
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小(文件夹里的图片的尺寸看起来不会有变化，只是之后分析的时候会临时变成这个尺寸)
        transforms.ToTensor(),  # 将图像转为Tensor才能被识别
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化(设置成大家经常用的参数)
    ])
    # 定义训练集和测试集
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    # shuffle=True:打乱读取顺序以提高效果
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 模型定义
def get_model():
    # 加载预训练的ResNet18模型(pretrained=True利用前人训练的经验)
    model = models.resnet18(pretrained=True)
    # model.fc.in_features指的是图片里的信息 告诉模型要把图片分成两类 如果只识别一种图像的话就设置成1
    model.fc = nn.Linear(model.fc.in_features, 2)   
    # 将模型移动到GPU（如果有GPU的话） cuda可利用显卡加快训练效率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model
