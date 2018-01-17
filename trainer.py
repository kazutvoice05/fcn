import numpy as np
import chainer
import chainer.function as F
import chainer.links as L
from chainer import cuda, optimizers, Variable
from chainer import training
from chainer.training import extensions
import cupy

import sys
import os
import argparse

from models.vgg16 import VGG16
from preprocess import load_data
from util import save_models

dataset_path = "/home/takagi/projects/dl_training/nyud_dataset/"
rgb_path = dataset_path + "rgb_images/"
label_path = dataset_path + "label_images/"
train_txt_path = dataset_path + "train.txt"

save_path = "/home/takagi/projects/dl_training/fcn/weights/first_vgg"

parser = argparse.ArgumentParser(description='Chainer VGG16 trainer')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--train_dataset', '-tr', default= rgb_path, type=str)
parser.add_argument('--target_dataset', '-ta', default= label_path, type=str)
parser.add_argument('--train_txt', '-tt', default= train_txt_path, type=str)
parser.add_argument('--batchsize', '-bm', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--epoch', '-e', default=300, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--classes', default=38, type=int) #37 + 1(background)
args = parser.parse_args()

n_epoch = args.epoch
n_class = args.classes
batchsize = args.batchsize
train_dataset = args.train_dataset
target_dataset = args.target_dataset
train_txt = args.train_txt

with open(train_txt,"r") as f:
    ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
n_data = len(names)
n_iter = n_data // batchsize
gpu_flag = True if args.gpu > 0 else False

model = VGG16(n_class)

if args.gpu >= 0:
    chainer.cuda._get_device(args.gpu).use()
    model.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy

optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

print ("## INFORMATION ##")
print ("Num Data: {}, Batchsize: {}, Iteration {}".format(n_data,batchsize,n_iter))

print("-"*40)

for epoch in range(1, n_epoch+1):
    print("epoch", epoch)
    for i in range(n_iter):

        model.cleargrads()
        indices = range(i * batchsize, (i+1)*batchsize)

        x = xp.zeros((batchsize, 3, 480, 640), dtype=xp.float32)
        y = xp.zeros((batchsize, 480, 640), dtype=xp.int32)

        for j in range(batchsize):
            name = names[i*batchsize + j]
            name = names[i * batchsize + j]
            xpath = train_dataset + name + ".png"
            ypath = target_dataset + name + ".png"
            x[j] = load_data(xpath, mode='data')
            y[j] = load_data(ypath, mode='label')

        x = Variable(x)
        y = Variable(y)
        loss = model(x, y, train = True)

        sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}".format(i+1, n_iter, loss.data))
        sys.stdout.flush()

        loss.backward()
        optimizer.update()

    if epoch % 50 == 0:
        save_models(model, optimizer, save_path, epoch)
    print("\n" + "-"*40)

save_models(model, optimizer, save_path, epoch)
