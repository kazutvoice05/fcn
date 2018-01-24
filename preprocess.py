import numpy as np
from PIL import Image
from chainer import cuda


def load_data(path, mode="label", xp=cuda.cupy):
    img = Image.open(path)
    simg = img.resize((224, 224))

    if mode == "label":
        y = xp.asarray(simg, dtype=xp.int32)

        return y

    elif mode == "data":
        x = xp.asarray(simg, dtype=xp.float32).transpose(2,0,1)
        return x

    elif mode == "predict":
        return simg
