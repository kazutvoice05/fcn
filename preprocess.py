import numpy as np
from PIL import Image
from chainer import cuda
import cupy


def load_data(path, mode="label", xp=cupy):
    img = Image.open(path)
    img = img.resize((320, 240))

    if mode == "label":
        y = xp.asarray(img, dtype=xp.int32)

        return y

    elif mode == "data":
        x = xp.asarray(img, dtype=xp.float32).transpose(2,0,1)
        return x

    elif mode == "predict":
        return img