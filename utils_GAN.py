import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
from os import listdir
from os.path import isfile,join

def kappa_my(confusion_matrix):
    num_img=[274,274,196,273,274,195,195,274,196,274]
    
    new_conf_mat=confusion_matrix
    
    sum_raw=new_conf_mat.sum(0)
    sim_col=new_conf_mat.sum(1)
    
    pe=(sum_raw*sim_col).sum()/sum(sim_col)/sum(sim_col)
    
    pa=0
    for j in range(len(num_img)):
        pa=pa+new_conf_mat[j,j]
    pa=pa/sum(sim_col)

    kappa=(pa-pe)/(1-pe)
    return kappa
    
def draw_mstar_confusion_matrix(cm,name):
    
    labels=['2S1','BRDM_2','BTR70','T62','ZIL131','BMP2','BTR60','D7','T72','ZSU234']
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cmap = plt.cm.binary
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=13, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                
                plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=13, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=13, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)

    plt.savefig(name+'.jpg', dpi=300)
#     plt.show()


######=======================================================================
def load_mstar_10(data_dir):
    
    def read_img(data_dir,root):
    
        img_rows, img_cols = 64, 64
        import collections
        from collections import defaultdict
        class_to_ix = {}
        ix_to_class = {}
        
        with open(data_dir+'/classes.txt', 'r') as txt:
            classes = [l.strip() for l in txt.readlines()]
            class_to_ix = dict(zip(classes, range(len(classes))))
#             print (class_to_ix)
            ix_to_class = dict(zip(range(len(classes)), classes))
#             print (ix_to_class)
            class_to_ix = {v: k for k, v in ix_to_class.items()}
        sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))
        all_imgs = []
        all_classes = []
        resize_count = 0
        invalid_count = 0
#         print(root)
        for i, subdir in enumerate(listdir(root)):
            imgs = listdir(join(root, subdir))
#             print(join(root, subdir))
            class_ix = class_to_ix[subdir]
            print(i, class_ix, subdir)
            for img_name in imgs:
                img_arr = plt.imread(join(root, subdir, img_name))
    #             print(img_arr.shape[0])
                if (img_arr.shape[0])>64:
                    img_arr_rs=np.zeros((img_arr.shape[0]/2,img_arr.shape[1]/2,1))
                    img_arr_rs[:,:,0] = img_arr[32:96,32:96]#[32:96,32:96]
#                     img_arr_rs[:,:,1] = img_arr[32:96,32:96]#[32:96,32:96]
#                     img_arr_rs[:,:,2] = img_arr[32:96,32:96]#[32:96,32:96]
                else:
                    img_arr_rs = img_arr#[:,:,0]#[32:96,32:96]

                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
                resize_count += 1
        print(len(all_imgs), 'images loaded')
        print(resize_count, 'images resized')
        print(invalid_count, 'images skipped')

        return np.array(all_imgs), np.array(all_classes)
    trX, trY =read_img(data_dir,data_dir+'/train')
    teX, teY =read_img(data_dir,data_dir+'/validation')
    
    trY = np.asarray(trY).astype(np.int)
#     teY = np.asarray(teY).astype(np.int)



    import random
    index_train = [i for i in range(len(trX))]
    random.shuffle(index_train)
    trX = trX[index_train,:,:,:]
    trY = trY[index_train]  
    
    index_train = [i for i in range(len(teX))]
    random.shuffle(index_train)
    teX = teX[index_train,:,:,:]
    teY = teY[index_train] 
    
    trY_vec = np.zeros((len(trY), 10), dtype=np.float)
    for i, label in enumerate(trY):
        trY_vec[i, trY[i]] = 1
    teY_vec=  teY
#     teY_vec = np.zeros((len(teY), 10), dtype=np.float)
#     for i, label in enumerate(teY):
#         teY_vec[i, teY[i]] = 1

    trX = trX.transpose(0, 3, 1, 2) / 255.
    teX = teX.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    trX = torch.from_numpy(trX).type(torch.FloatTensor)
    trY_vec = torch.from_numpy(trY_vec).type(torch.FloatTensor)
    
    teX = torch.from_numpy(teX).type(torch.FloatTensor)
#     teY_vec = torch.from_numpy(teY_vec).type(torch.FloatTensor)
    return trX,teX,trY_vec,teY_vec
######=======================================================================

def load_mstar_11class(data_dir):
    
    def read_img(data_dir,root):
    
        img_rows, img_cols = 64, 64
        import collections
        from collections import defaultdict
        class_to_ix = {}
        ix_to_class = {}
        with open(data_dir+'/classes.txt', 'r') as txt:
            classes = [l.strip() for l in txt.readlines()]
            class_to_ix = dict(zip(classes, range(len(classes))))
#             print (class_to_ix)
            ix_to_class = dict(zip(range(len(classes)), classes))
#             print (ix_to_class)
            class_to_ix = {v: k for k, v in ix_to_class.items()}
        sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))
        all_imgs = []
        all_classes = []
        resize_count = 0
        invalid_count = 0
#         print(root)
        for i, subdir in enumerate(listdir(root)):
            imgs = listdir(join(root, subdir))
#             print(join(root, subdir))
            class_ix = class_to_ix[subdir]
            print(i, class_ix, subdir)
            for img_name in imgs:
                img_arr = plt.imread(join(root, subdir, img_name))
    #             print(img_arr.shape[0])
                if (img_arr.shape[0])>64:
                    img_arr_rs=np.zeros((img_arr.shape[0]/2,img_arr.shape[1]/2,1))
                    img_arr_rs[:,:,0] = img_arr[32:96,32:96]#[32:96,32:96]
#                     img_arr_rs[:,:,1] = img_arr[32:96,32:96]#[32:96,32:96]
#                     img_arr_rs[:,:,2] = img_arr[32:96,32:96]#[32:96,32:96]
                else:
                    img_arr_rs = img_arr#[:,:,0]#[32:96,32:96]

                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
                resize_count += 1
        print(len(all_imgs), 'images loaded')
        print(resize_count, 'images resized')
        print(invalid_count, 'images skipped')

        return np.array(all_imgs), np.array(all_classes)
    trX, trY =read_img(data_dir,data_dir+'/train')
#     teX, teY =read_img('./data/mstar_cnn/validation')
    
    trY = np.asarray(trY).astype(np.int)
#     teY = np.asarray(teY)

    X=trX#= np.concatenate((trX, teX), axis=0)
    y=trY# = np.concatenate((trY, teY), axis=0).astype(np.int)

    import random
    index_train = [i for i in range(len(trX))]
    random.shuffle(index_train)
    X = X[index_train,:,:,:]
    y = y[index_train]  
    

    y_vec = np.zeros((len(y), 11), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec
##########=========================================================

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
#     print(dir)
    dset = datasets.ImageFolder(dir, transform)
#     dsets=dset[:][:][:][:][0]
#     print(dset.type)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path1 = os.path.join(path, model_name + '_D_loss.png')

    plt.savefig(path1)
#=====================
#     x = range(len(hist['D2_loss']))

#     y1 = hist['D2_loss']
#     y2 = hist['G_loss']

#     plt.plot(x, y1, label='D_loss')
#     plt.plot(x, y2, label='G_loss')

#     plt.xlabel('Iter')
#     plt.ylabel('Loss')

#     plt.legend(loc=4)
#     plt.grid(True)
#     plt.tight_layout()

#     path2 = os.path.join(path, model_name + '_D2_loss.png')

#     plt.savefig(path2)
    
    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            try:
                m.bias.data.zero_()
            except:
                print("there is no bias")