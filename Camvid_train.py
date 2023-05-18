import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.camvid_data import CamvidDataset
from model.VGG16 import VGG16


# 超参数
num_classes = 32
batch_size = 8
learning_rate = 0.001
num_epochs = 10

root_dir = 'dataset/Camvid'
train_image_folder = 'train'
train_label_folder = 'train_labels'

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



# train_dataset = Camvid(root='./dataset', split='train', download=True, transform=transform)
train_dataset = CamvidDataset(root_dir, train_image_folder, train_label_folder, transform)
# test_dataset = Camvid(root='./dataset', split='test', download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 实例化VGG16模型
model = VGG16(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 将模型移至GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# # 在测试集上评估模型
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#
#         outputs = model(images)
#         _, predicted = torch.max(outputs.dataset, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('在测试集上的准确率: {}%'.format(100 * correct / total))

