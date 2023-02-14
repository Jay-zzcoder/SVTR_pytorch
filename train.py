import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import sys
import os
import argparse
import random
import time
import datetime
from tqdm import tqdm
from SVTR import SVTR
from utils.str_process import strLabelConverter
from dataset import ICDAR2015_wordrecognition


parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', default="./data/training_word_images_gt/",  help='path to dataset')
parser.add_argument('--text_path', default="./data/training_word_images_gt/gt.txt",  help='path to text')
parser.add_argument('--valRoot', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--device', default="cuda", action='store_true', help='enables cuda')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
arg = parser.parse_args()
print(arg)

random.seed(arg.manualSeed)
np.random.seed(arg.manualSeed)
torch.manual_seed(arg.manualSeed)

trans =transforms.Compose([transforms.Resize([32, 100]), transforms.ToTensor()])

def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)

def train(dataloader, model, optimizer, criterion, scheduler):
    converter = strLabelConverter(arg.alphabet)
    model.train()
    text = torch.IntTensor(arg.batch_size * 5)
    length = torch.IntTensor(arg.batch_size)
    text = Variable(text)
    length = Variable(length)
    print(f"Start training for {arg.epochs} epochs")
    start_time = time.time()
    current_loss = 0
    min_loss = sys.maxsize
    for i in range(arg.epochs):
        running_loss = 0
        for j, data in enumerate(tqdm(dataloader)):
            img, tt = data
            img = img.to(arg.device)

            t, l = converter.encode(tt)
            loadData(text, t)
            loadData(length, l)
            preds = model(img)
            batch_size = img.size(0)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            # print("preds_size shape: ", preds_size.shape)
            # print("predse shape: ", preds.shape)
            # print("text shape: ", text.shape)
            loss = criterion(preds.float(), text, preds_size, length) / batch_size
            if not np.isnan(loss.item()) and not loss.item() == float("inf"):
                running_loss = running_loss + loss.item()

            # _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            # preds = preds.transpose(1, 0).contiguous().view(-1)
            # sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"the running_loss of {i} epoch is {running_loss}")
        if current_loss < min_loss:
            current_loss = min_loss
            torch.save(model.state_dict(), arg.save_path + "minloss.pth")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



def main():
    train_datasets = ICDAR2015_wordrecognition(arg.trainRoot, arg.text_path, transformes=trans)
    train_dataloader = DataLoader(train_datasets, train=True, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    model = SVTR().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CTCLoss().cuda()
    train(train_dataloader, model, optimizer, criterion, scheduler)


if __name__ == "__main__":
    #main()
    converter = strLabelConverter(arg.alphabet)
    t, l = converter.encode(("book", "cat"))
    print(t)
    print(l)











