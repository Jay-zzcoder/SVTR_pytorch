import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from torchvision import transforms
from SVTR import SVTR
from utils.str_process import strLabelConverter
from dataset import ICDAR2015_wordrecognition
import os
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--text_path', default="./data/training_word_images_gt/gt.txt",  help='path to text')
parser.add_argument('--model_path', default="./model/training_word_images_gt/gt.txt",  help='path to text')
parser.add_argument('--model_path', default="./results/",  help='path to text')
parser.add_argument('--valRoot', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--device', default="cuda", action='store_true', help='enables cuda')
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





def eval(test_path):
    pass


transformer = transforms.Compose([transforms.Resize([32, 100]), transforms.ToTensor()])
converter = strLabelConverter(arg.alphabet)

def inference(path, save_path):
    model = SVTR()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(arg.model_path))
    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith(".png"):
                continue
            img_path = root + file
            image = Image.open(img_path)
            image = transformer(image)
            if torch.cuda.is_available():
                image = image.cuda()
            image = image.unsqueeze(0)
            model.eval()
            preds = model(image)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            cv2.putText(image, sim_pred, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.write(save_path+file, image)


test_path = "./data/test_word_images_gt/"
if __name__ =="__main__":
    inference(arg.test_path, arg.save_path)








