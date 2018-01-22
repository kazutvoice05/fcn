import os.path as osp

import chainer
import chainer.functions as F
import chainer.links as L


class VGG16(chainer.Chain):

    pretrained_model = osp.expanduser('/home/takagi/projects/dl_training/vgg16_from_caffe.npz')

    def __init__(self, n_class=38):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)

            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)

            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)

            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)

            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)

            self.fc6 = L.Linear(25088, 4096)
            #self.fc6 = L.Linear(153600, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, n_class)

    def __call__(self, x, t=None, train=False, test=False):
        h = x
        c11= F.relu(self.conv1_1(h))
        print("c11")
        print(c11.shape)
        c12 = F.relu(self.conv1_2(c11))
        print("c12")
        print(c12.shape)
        p1 = F.max_pooling_2d(c12, 2, stride=2)
        print("p1")
        print(p1.shape)

        c21 = F.relu(self.conv2_1(p1))
        print("c21")
        print(c21.shape)
        c22 = F.relu(self.conv2_2(c21))
        print("c22")
        print(c22.shape)
        p2 = F.max_pooling_2d(c22, 2, stride=2)
        print("p2")
        print(p2.shape)

        c31 = F.relu(self.conv3_1(p2))
        print("c311")
        print(c31.shape)
        c32 = F.relu(self.conv3_2(c31))
        print("c32")
        print(c32.shape)
        c33 = F.relu(self.conv3_3(c32))
        print("c33")
        print(c33.shape)
        p3 = F.max_pooling_2d(c33, 2, stride=2)
        print("p3")
        print(p3.shape)

        c41 = F.relu(self.conv4_1(p3))
        print("c41")
        print(c41.shape)
        c42 = F.relu(self.conv4_2(c41))
        print("c42")
        print(c42.shape)
        c43 = F.relu(self.conv4_3(c42))
        print("c43")
        print(c43.shape)
        p4 = F.max_pooling_2d(c43, 2, stride=2)
        print("p4")
        print(p4.shape)

        c51 = F.relu(self.conv5_1(p4))
        print("c51")
        print(c51.shape)
        c52 = F.relu(self.conv5_2(c51))
        print("c52")
        print(c52.shape)
        c53 = F.relu(self.conv5_3(c52))
        print("c53")
        print(c53.shape)
        p5 = F.max_pooling_2d(c53, 2, stride=2)
        print("p5")
        print(p5.shape)

        f6 = F.dropout(F.relu(self.fc6(p5)), ratio=.5)
        f7 = F.dropout(F.relu(self.fc7(f6)), ratio=.5)
        f8 = self.fc8(f7)

        self.score = f8

        if t is None:
            assert not chainer.config.train
            return

        if train:
            self.loss = F.softmax_cross_entropy(f8, t)
            self.accuracy = F.accuracy(self.score, t)
            return self.loss
        else:
            pred = F.softmax(f8)
            return pred

