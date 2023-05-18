import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

#1
class CamvidDataset(Dataset):
    def __init__(self, root_dir, image_folder, label_folder, transform=None):
        self.root_dir = root_dir
        self.image_folder = image_folder
        self.label_folder = label_folder
        # self.transform = transform
        self.transform = ToTensor()

        self.image_path_list = os.listdir(os.path.join(root_dir, image_folder))
        self.label_path_list = os.listdir(os.path.join(root_dir, label_folder))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_folder, self.image_path_list[index])
        label_path = os.path.join(self.root_dir, self.label_folder, self.label_path_list[index])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)
            # label = self.transform(label)
            label = self.transform(label)  # 对标签应用转换

        return image, label
