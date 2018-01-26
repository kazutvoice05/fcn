import os
from chainer import serializers


def save_models(model, optimizer, path, epoch):
    if not os.path.exists(path +"/epoch_{}".format(epoch)):
        dir_path = path +"/epoch_{}".format(epoch)
        os.mkdir(dir_path)
    serializers.save_npz(dir_path + "/chainer_vgg.weight", model)
    serializers.save_npz(dir_path + "/chainer_vgg.state", optimizer)
    print('save weight in epoch {}'.format(epoch))

def update_datas(accuracy, loss, path):
    assert type(accuracy) == str
    assert type(loss) == str

    f_path = path + "data.csv"
    if not os.path.exists(f_path):
        f = open(f_path,'w')
        f.writelines("accuracy,loss\n")
        f.writelines(accuracy + "," + loss + "\n")
        f.close()
    else:
        f = open(f_path,'a')
        f.writelines(accuracy + "," + loss + "\n")
        f.close()
