import torch
from utils import get_data_loaders, get_model
import matplotlib.pyplot as plt
import numpy as np

# 反向归一化处理，以便显示图片
def imshow(img):
    # 反归一化 将数据从 [-1, 1] 转回好处理的[0, 1]
    img = img / 2 + 0.5  
    # clip:限制数值范围
    img = img.clip(0, 1)     
    npimg = img.cpu().numpy()  # 将张量移动到CPU 上,再转为 NumPy 数组  无法直接从GPU转换 
    # 转换维度:imshow 需要数据的形状为 (height, width, channels)，而 PyTorch 的tensor通常是 (channels, height, width)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 测试模型
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    # no_grad:不用计算
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main():
    # 获取数据加载器
    _, test_loader = get_data_loaders()

    # 获取模型和设备
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载已训练的模型
    model.load_state_dict(torch.load('cat_dog_model.pth'))
    model = model.to(device)

    # 评估模型性能
    test_accuracy = test(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

   # 可视化结果 获取第一个批次的数据 iter:将对象转换为迭代器
    dataiter = iter(test_loader)
    images, labels = next(dataiter)   

    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 显示图片及其预测结果
    print(f"真实标签: {labels}")
    print(f"预测标签: {predicted}")

    # 显示第一张图像并标明真实和预测标签
    imshow(images[0])  # 显示这批里的第一张图像
    first_prediction = predicted[0].item()
    first_label = labels[0].item()
    print(f'first_prediction: {first_prediction}')
    print(f'first_label: {first_label}')

if __name__ == "__main__":
    main()