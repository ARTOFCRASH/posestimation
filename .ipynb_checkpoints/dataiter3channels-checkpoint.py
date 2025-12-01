from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import numpy as np


class MyData(Dataset):
    def __init__(self, img_dir, trans=None, roll=False, pitch=False):
        self.img_dir = img_dir
        self.transform = trans
        self.imgs = []
        for f in os.listdir(img_dir):
            # 如果 f 是 bytes，就 decode 成 str
            if isinstance(f, bytes):
                f = f.decode('utf-8', errors='ignore')  # 或根据实际编码指定 errors='strict'
            # 然后再判断后缀
            if f.endswith('.png'):
                self.imgs.append(f)
        self.roll = roll
        self.pitch = pitch

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 从文件名中提取标签
        label_values = img_name.split('.')[0].split('_')[:2]
        roll, pitch = map(float, label_values)
        normalized_roll = np.interp(roll, [-50, 50], [-1, 1])
        normalized_pitch = np.interp(pitch, [-50, 50], [-1, 1])

        if self.transform:
            image = self.transform(image)
        labels = []
        if self.roll:
            labels.append(roll)
        if self.pitch:
            labels.append(pitch)

        depth = image[2:3, :, :]
        third_channel = image[0:1, :, :]
        color_channel = torch.cat([image[0:2, :, :], third_channel], dim=0)

        return color_channel, depth, torch.tensor(labels, dtype=torch.float32)


if __name__ == '__main__':
    # 定义转换操作
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.5699, 0.4200, 0.3462), (0.3303, 0.2403, 0.2773))
    ])

    # 创建数据集实例
    dataset = MyData('/root/autodl-tmp/dataset/images/train', trans=transform, roll=True, pitch=True)
    print(f'length of dataset: {len(dataset)}')
    _, labels = dataset[20]
    print(f'{labels}')
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for images, labels in train_dataloader:
        print(labels)
        break
