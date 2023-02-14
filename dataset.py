from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import deeplake
from torchvision import transforms
from PIL import Image



root_path = "./data/training_word_images_gt/"
text_path = "./data/training_word_images_gt/gt.txt"

class ICDAR2015_wordrecognition(Dataset):
    def __init__(self, root_path, text_path, transformes=None, encoder=None):
        super(ICDAR2015_wordrecognition, self).__init__()
        self.root_path = root_path
        self.text_path = text_path
        self.data = self.get_data()
        self.transform = transformes
        self.encoder = encoder

    def __getitem__(self, index):
        data_pair = self.data[index]
        img_path, text = data_pair
        print(type(text))
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.encoder is not None:
            text = self.encoder(text)

        return (np.array(img), text)


    def __len__(self):
        return len(self.data)

    def get_data(self):
        data = []
        for line in open(self.text_path, "r"):
            path = line.split(", ")[0]
            s = line.split(", ")[1][1:-2]
            text = ''.join(filter(str.isalnum, s))
            imgpath = self.root_path + path
            data.append([imgpath, text])
        return data


if __name__ == "__main__":
    trans = transforms.Compose([ transforms.CenterCrop(10),
    transforms.ToTensor()])

    datasets= ICDAR2015_wordrecognition(root_path, text_path)
    dataloder = DataLoader(datasets, batch_size=1, shuffle=False)
    for data in dataloder:
        img, text = data
        if not isinstance(text, str):
            text = text[0]
        break




























