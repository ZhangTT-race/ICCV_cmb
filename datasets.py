import torch
from PIL import Image
from torch.autograd import Variable
import os


class FASDataset(torch.utils.data.Dataset):
    def __init__(self, label_path, img_dir="", transform=None, phase="train"):
        super(FASDataset, self).__init__()

        self.phase = phase

        imgs = []
        # print("img dir:{}".format(img_dir))
        # print("label path:{}".format(label_path))
        a_label_path = os.path.join(img_dir, label_path)
        # print("join path:{}".format(a_label_path))
        with open(a_label_path, 'r') as T:
            for line in T.readlines():
                line = line.strip()
                r_path, label = line.split(" ")
                a_path = os.path.join(img_dir, "phase1", r_path)
                imgs.append((a_path, int(label)))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, item):
        path, label = self.imgs[item]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.phase == "test":
            return img, label, path
        return img, label

    def __len__(self):
        return len(self.imgs)
