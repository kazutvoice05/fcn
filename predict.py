#coding: 'utf-8'

"""
my_fcn
predict

created by Kazunari on 2017/12/09 
"""

from chainer import serializers
import numpy as np
from PIL import Image
import os
import argparse
import cv2
import cupy

from models.fcn32s import FCN32s
from preprocess import load_data
from color_map import make_color_map


parser = argparse.ArgumentParser(description='Chainer Fully Convolutional Network: predict')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--image_path', '-i', default='/home/takagi/projects/dl_training/nyud_dataset/rgb_images/0.png', type=str)
parser.add_argument('--weight', '-w', default="weight/chainer_fcn.weight", type=str)
parser.add_argument('--classes', default=38, type=int)
#parser.add_argument('--classes', default=21, type=int)
args = parser.parse_args()

xp = cupy if args.gpu >= 0 else np

img_name = args.image_path.split("/")[-1].split(".")[0]

color_map = make_color_map(args.classes) #要仕様変更
model = FCN32s(n_class= args.classes)
#serializers.load_npz('weight/chainer_fcn.weight', model)
serializers.load_npz(args.weight, model)

o = load_data(args.image_path, args.gpu, mode="predict")
x = load_data(args.image_path, args.gpu, mode="data")
x = xp.expand_dims(x, axis=0)
pred = model.predict(x)
pred = pred[0]

row, col = pred.shape
dst = xp.ones((row, col, 3))
for i in range(args.classes):
  dst[pred == i] = color_map[i]
img = Image.fromarray(xp.uint8(dst))

b,g,r = img.split()
img = Image.merge("RGB",(r,g,b))

trans = Image.new('RGBA', img.size, (0,0,0,0))
w, h = img.size
for x in range(w):
  for y in range(h):
    pixel = img.getpixel((x,y))
    if (pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0) or (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
      continue
    trans.putpixel((x,y), pixel)

if not os.path.exists("out"):
  os.mkdir("out")
o.save("out/original.png")
trans.save("out/pred.png")

o = cv2.imread("out/original.png", 1)
p = cv2.imread("out/pred.png", 1)

pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

cv2.imwrite("out/pred_{}.png".format(img_name), pred)


