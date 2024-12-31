import torch
from PIL import Image
from utils import get_model
from torchvision import transforms

def main():
    # 设置路径和模型
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载已训练的模型
    model.load_state_dict(torch.load('cat_dog_model.pth'))
    model = model.to(device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 对一张图片进行预测
    def predict_image(image_path):
        img = Image.open(image_path)
        # unsqueeze:在tensor的第一个位置插入一个新的维度   
        # 在深度学习中，通常输入到神经网络的数据是以批次(4个维度)的形式传入 但是单张照片只有3个维度，所以要加一个
        img = transform(img).unsqueeze(0)  # 增加批次维度

        img = img.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        return "Dog" if predicted.item() == 1 else "Cat"

    # 测试预测
    image_path = r'D:/xiao/code/CatVsDog/predict/19.jpg'
    prediction = predict_image(image_path)
    print(f'The image is a {prediction}')

if __name__ == "__main__":
    main()