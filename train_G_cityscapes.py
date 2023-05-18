import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes


class VGG16Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Segmentation, self).__init__()
        self.vgg16 = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())[:-1])
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(512, num_classes, kernel_size=32, stride=32)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.conv(x)
        x = F.relu(x)
        x = self.deconv(x)
        x = x[:, :, 3:35, 6:38]  # cropping to match the input size
        return x

transform = transforms.Compose([
    transforms.RandomCrop((256, 512)),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ToTensor(),
])

# 定义了一个collate_fn函数，它接受一个批次的图像和标签，并将它们转换为张量
def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    # 将PIL Image对象转换为张量
    images = torch.stack([transforms.ToTensor()(img) for img in images])
    labels = torch.stack([torch.Tensor(np.array(label)) for label in labels])

    return images, labels

# ,collate_fn=collate_fn
train_set = Cityscapes(root='./dataset/cityscapes', split='train', mode='fine', target_type='semantic', transform=None)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True,collate_fn=collate_fn)


model = VGG16Segmentation(num_classes=20)  # Assuming Cityscapes has 20 classes
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
# key = 2000

for epoch in range(num_epochs):
    model.train()



    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')





