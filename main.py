
# coding: utf-8


# # Step 1 : read data and train the target CGAN (TCGAN)

# ## 1.1 read training data and test data from a large scale SAR image

import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import sklearn
from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage import data,color,morphology,measure


iput_img_original=image.imread('../../datasets/fangchenggang/A2_广西防城港.jpg')
gt_original=image.imread('../../datasets/fangchenggang/ground_truth2.jpg')

h_ipt=gt_original.shape[0]
w_ipt=gt_original.shape[1]

# input the parameter of the data
patch_size=16
num_label=5
n_segments=8000
num_train=18


# crop the original input to a set of smaller size images
rownum=6
colnum=6
train_patch_ind=[5,10,15,20,21,25,30]

#定义一个函数，按照super-pixel的质心，把super pixel截取下来，放到一个patch里面。
#输入为superpixel的质心，patch的size,和一个padding之后的输入图像

def crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size):
    patch_out=new_input_padding[wi_ipt:wi_ipt+patch_size,hi_ipt:hi_ipt+patch_size]
    return patch_out


def read_img(rownum,colnum,iput_img_original,gt_original,train_patch_ind):
    ALL_DATA_X_L=[]
    ALL_DATA_Y_L=[]
    
    ALL_DATA_X_no_L=[]
    ALL_DATA_Y_no_L=[]    
    
    rowheight = h_ipt // rownum
    colwidth = w_ipt // colnum

    for r in range(rownum):#
        for c in range(colnum):#           
            iput_img= iput_img_original[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth];
            gt      = gt_original      [r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth,:];
          
            segments = slic(iput_img, n_segments=n_segments, compactness=0.5)
            out=mark_boundaries(gt,segments)
#             plt.figure(dpi=200)
#             plt.imshow(out)
#             plt.show()
            #得到每个super pixel的质心
            segments_label=segments+1  #这里+1，是因为regionprops函数的输入要求是label之后的图像，而label的图像的区域编号是从1开始的
            region_fea=measure.regionprops(segments_label)

            #定义一个边界扩大的patch_size的空矩阵，主要是为了当super pixel位于图像边缘时，    
            new_input_padding=np.zeros([rowheight+patch_size,colwidth+patch_size])
            #把这个iput_img放到new_input_padding中
            new_input_padding[patch_size/2:-patch_size/2,patch_size/2:-patch_size/2]=iput_img
            
            print( r*rownum+c )
            if r*rownum+c in train_patch_ind:          
                 #对所有的super pixel开始循环
                for ind_pixel in range (segments_label.max()):

                    #计算当前superpixel的质心，为了生成切片，切片以这个质心为中心
                    centriod=np.array(region_fea[ind_pixel].centroid).astype("int32")
                    wi_ipt=centriod[0]
                    hi_ipt=centriod[1]

                    #得到这个超像素的所有像素的坐标，根据坐标能够知道这个超像素在GT图中的所有像素值all_pixels
                    #根据所有的像素，得到哪一个像素值最多，例如【0,0,0】最多，那这个超像素的标签就是“河流”

                    all_pixels=gt[region_fea[ind_pixel].coords[:,0],region_fea[ind_pixel].coords[:,1]]
                    n0 = np.bincount(all_pixels[:,0])
                    n1 = np.bincount(all_pixels[:,1])  
                    n2 = np.bincount(all_pixels[:,2])  
                    gt_of_superp=[n0.argmax(),n1.argmax(),n2.argmax()] #gt_of_superp这个超像素中出现最多次的像素值


                    if gt_of_superp[0]<=20 and gt_of_superp[1]>=240 and gt_of_superp[2]<=20:  
                        ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_L.append(0)

                    # black ---river 
                    elif gt_of_superp[0]<=50 and gt_of_superp[1]<=50 and gt_of_superp[2]<=50: 
                        ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_L.append(1)

                    # red ---urban area 
                    elif gt_of_superp[0]>=200 and gt_of_superp[1]<=50 and gt_of_superp[2]<=50: 
                        ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_L.append(2)

                    # yellow --- framland 
                    elif gt_of_superp[0]>=200 and gt_of_superp[1]>=200 and gt_of_superp[2]<=50: 
                        ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_L.append(3)

                        # blue ---road
                    elif gt_of_superp[0]<=50 and gt_of_superp[1]<=50 and gt_of_superp[2]>=200:  
                        ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_L.append(2)

                    # white ---background
                    elif gt_of_superp[0]>=200 and gt_of_superp[1]>=200 and gt_of_superp[2]>=200:  
                        ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_L.append(4)


                    # other color regions are regarded as the background
                    else:
                        ALL_DATA_X_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_L.append(4)
                        
            else:
                for ind_pixel in range (segments_label.max()):

                    #计算当前superpixel的质心，为了生成切片，切片以这个质心为中心
                    centriod=np.array(region_fea[ind_pixel].centroid).astype("int32")
                    wi_ipt=centriod[0]
                    hi_ipt=centriod[1]

                    #得到这个超像素的所有像素的坐标，根据坐标能够知道这个超像素在GT图中的所有像素值all_pixels
                    #根据所有的像素，得到哪一个像素值最多，例如【0,0,0】最多，那这个超像素的标签就是“河流”

                    all_pixels=gt[region_fea[ind_pixel].coords[:,0],region_fea[ind_pixel].coords[:,1]]
                    n0 = np.bincount(all_pixels[:,0])
                    n1 = np.bincount(all_pixels[:,1])  
                    n2 = np.bincount(all_pixels[:,2])  
                    gt_of_superp=[n0.argmax(),n1.argmax(),n2.argmax()] #gt_of_superp这个超像素中出现最多次的像素值


                    if gt_of_superp[0]<=20 and gt_of_superp[1]>=240 and gt_of_superp[2]<=20:  
                        ALL_DATA_X_no_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_no_L.append(0)

                    # black ---river 
                    elif gt_of_superp[0]<=50 and gt_of_superp[1]<=50 and gt_of_superp[2]<=50: 
                        ALL_DATA_X_no_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_no_L.append(1)

                    # red ---urban area 
                    elif gt_of_superp[0]>=200 and gt_of_superp[1]<=50 and gt_of_superp[2]<=50: 
                        ALL_DATA_X_no_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_no_L.append(2)

                    # yellow --- framland 
                    elif gt_of_superp[0]>=200 and gt_of_superp[1]>=200 and gt_of_superp[2]<=50: 
                        ALL_DATA_X_no_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_no_L.append(3)

                        # blue ---road
                    elif gt_of_superp[0]<=50 and gt_of_superp[1]<=50 and gt_of_superp[2]>=200:  
                        ALL_DATA_X_no_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_no_L.append(2)

                    # white ---background
                    elif gt_of_superp[0]>=200 and gt_of_superp[1]>=200 and gt_of_superp[2]>=200:  
                        ALL_DATA_X_no_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_no_L.append(4)
                        
                    # other color regions are regarded as the background
                    else:
                        ALL_DATA_X_no_L.append(crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size))
                        ALL_DATA_Y_no_L.append(4) 
                        

    return np.array(ALL_DATA_X_L).astype(np.int),  np.array(ALL_DATA_Y_L),  np.array(ALL_DATA_X_no_L) ,np.array(ALL_DATA_Y_no_L)




ALL_DATA_X_L, ALL_DATA_Y_L, ALL_DATA_X_no_L,ALL_DATA_Y_no_L=read_img(rownum,colnum,iput_img_original,gt_original,train_patch_ind)


import random
index_train = [i for i in range(len(ALL_DATA_X_L))]
random.shuffle(index_train)
ALL_DATA_X_L = ALL_DATA_X_L[index_train,:,:]
ALL_DATA_X_L = ALL_DATA_X_L.reshape(ALL_DATA_X_L.shape[0],1,ALL_DATA_X_L.shape[1],ALL_DATA_X_L.shape[2])/ 255.
ALL_DATA_X_L = torch.from_numpy(ALL_DATA_X_L).type(torch.FloatTensor)


ALL_DATA_Y_L = ALL_DATA_Y_L[index_train]  
ALL_DATA_Y_L_vec = np.zeros((len(ALL_DATA_Y_L), num_label), dtype=np.float)
for i, label in enumerate(ALL_DATA_Y_L):
    ALL_DATA_Y_L_vec[i, ALL_DATA_Y_L[i]] = 1
ALL_DATA_Y_L_vec = torch.from_numpy(ALL_DATA_Y_L_vec).type(torch.FloatTensor)



import random
index_train = [i for i in range(len(ALL_DATA_X_no_L))]
random.shuffle(index_train)
ALL_DATA_X_no_L = ALL_DATA_X_no_L[index_train,:,:]
ALL_DATA_X_no_L = ALL_DATA_X_no_L.reshape(ALL_DATA_X_no_L.shape[0],1,ALL_DATA_X_no_L.shape[1],ALL_DATA_X_no_L.shape[2])/ 255.
ALL_DATA_X_no_L = torch.from_numpy(ALL_DATA_X_no_L).type(torch.FloatTensor)


ALL_DATA_Y_no_L = ALL_DATA_Y_no_L[index_train]  



# ## TCGAN网络参数初始化

import argparse, os
import numpy as np

import utils_GAN
"""parsing and configuration"""

class Config:
    epoch=int(30)
    batch_size=int(330)
    save_dir='models'
    result_dir='results'
    log_dir='logs'
    lrG=float(0.0002)
    lrD=float(0.0002)
    beta1=float(0.5)
    beta2 =float(0.999)
    gpu_mode=True
    
args=Config()

print("finished")


# ## train TCGAN

from DCGAN_FangChengGang_16 import DCGAN_1G_Class_MSTAR_16
import os

print(" [*] train the 1G-class GANs ")
args.gan_type= 'DCGAN_1G_Class_MSTAR'
gan_1G = DCGAN_1G_Class_MSTAR_16(args,ALL_DATA_X_L, ALL_DATA_Y_L_vec, ALL_DATA_X_no_L)
gan_1G.train()

print("finished")


########################################################################################################

# # Step 2. Train BCGAN

# ## 读取标记数据和未标记数据<div class="cite2c-biblio"></div>

import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import sklearn
from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage import data,color,morphology,measure


iput_img_original=image.imread('../../datasets/fangchenggang/A2_广西防城港.jpg')
gt_original=image.imread('../../datasets/fangchenggang/ground_truth2.jpg')

h_ipt=gt_original.shape[0]
w_ipt=gt_original.shape[1]

# input the parameter of the data
patch_size=64
num_label=5
n_segments=8000
num_train=18


# crop the original input to a set of smaller size images
rownum=6
colnum=6
train_patch_ind=[5,10,15,20,21,25,30]


#定义一个函数，按照super-pixel的质心，把super pixel截取下来，放到一个patch里面。
#输入为superpixel的质心，patch的size,和一个padding之后的输入图像

def crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size):
    patch_out=new_input_padding[wi_ipt:wi_ipt+patch_size,hi_ipt:hi_ipt+patch_size]
    return patch_out


ALL_DATA_X_L, ALL_DATA_Y_L, ALL_DATA_X_no_L,ALL_DATA_Y_no_L=read_img(rownum,colnum,iput_img_original,gt_original,train_patch_ind)

import random
index_train = [i for i in range(len(ALL_DATA_X_L))]
random.shuffle(index_train)
ALL_DATA_X_L = ALL_DATA_X_L[index_train,:,:]
ALL_DATA_X_L = ALL_DATA_X_L.reshape(ALL_DATA_X_L.shape[0],1,ALL_DATA_X_L.shape[1],ALL_DATA_X_L.shape[2])/ 255.
ALL_DATA_X_L = torch.from_numpy(ALL_DATA_X_L).type(torch.FloatTensor)


ALL_DATA_Y_L = ALL_DATA_Y_L[index_train]  
ALL_DATA_Y_L_vec = np.zeros((len(ALL_DATA_Y_L), num_label), dtype=np.float)
for i, label in enumerate(ALL_DATA_Y_L):
    ALL_DATA_Y_L_vec[i, ALL_DATA_Y_L[i]] = 1
ALL_DATA_Y_L_vec = torch.from_numpy(ALL_DATA_Y_L_vec).type(torch.FloatTensor)



import random
index_train = [i for i in range(len(ALL_DATA_X_no_L))]
random.shuffle(index_train)
ALL_DATA_X_no_L = ALL_DATA_X_no_L[index_train,:,:]
ALL_DATA_X_no_L = ALL_DATA_X_no_L.reshape(ALL_DATA_X_no_L.shape[0],1,ALL_DATA_X_no_L.shape[1],ALL_DATA_X_no_L.shape[2])/ 255.
ALL_DATA_X_no_L = torch.from_numpy(ALL_DATA_X_no_L).type(torch.FloatTensor)


ALL_DATA_Y_no_L = ALL_DATA_Y_no_L[index_train]  


# ## BCGAN网络参数初始化



import argparse, os
import numpy as np


import utils_GAN
"""parsing and configuration"""

class Config:
    epoch=int(30)
    batch_size=int(330)
    save_dir='models'
    result_dir='results'
    log_dir='logs'
    lrG=float(0.0002)
    lrD=float(0.0002)
    beta1=float(0.5)
    beta2 =float(0.999)
    gpu_mode=True
    
args=Config()

print("finished")


# ## Train BCGAN

from DCGAN_FangChengGang_64 import DCGAN_1G_Class_MSTAR_64
import os
print(" [*] train the 1G-class GANs ")
args.gan_type= 'DCGAN_1G_Class_MSTAR'
gan_1G = DCGAN_1G_Class_MSTAR_64(args,ALL_DATA_X_L, ALL_DATA_Y_L_vec, ALL_DATA_X_no_L)
gan_1G.train()

print("finished")




####################################################################################################3
# # Fine-tune the hierarchical CGAN

import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import sklearn
from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage import data,color,morphology,measure
from Read_img_Gan import Read_img_Gan

iput_img_original=image.imread('../../datasets/fangchenggang/A2_广西防城港.jpg')
gt_original=image.imread('../../datasets/fangchenggang/ground_truth2.jpg')

h_ipt=gt_original.shape[0]
w_ipt=gt_original.shape[1]
patch_size=16
num_label=5
n_segments=8000
num_train=18
rownum=6
colnum=6
train_patch_ind=[5,10,15,20,21,25,30]

ALL_DATA_X_L_small, ALL_DATA_Y_L_small, ALL_DATA_X_no_L_small,ALL_DATA_Y_no_L_small=Read_img_Gan(h_ipt,w_ipt,n_segments,patch_size,num_label,rownum,colnum,iput_img_original,gt_original,train_patch_ind)

import random
index_train = [i for i in range(len(ALL_DATA_X_L_small))]
random.shuffle(index_train)
ALL_DATA_X_L_small = ALL_DATA_X_L_small[index_train,:,:]
ALL_DATA_X_L_small = ALL_DATA_X_L_small.reshape(ALL_DATA_X_L_small.shape[0],1,ALL_DATA_X_L_small.shape[1],ALL_DATA_X_L_small.shape[2])/ 255.
ALL_DATA_X_L_small = torch.from_numpy(ALL_DATA_X_L_small).type(torch.FloatTensor)

ALL_DATA_Y_L_small = ALL_DATA_Y_L_small[index_train]  
ALL_DATA_Y_L_vec_small = np.zeros((len(ALL_DATA_Y_L_small), num_label), dtype=np.float)
for i, label in enumerate(ALL_DATA_Y_L_small):
    ALL_DATA_Y_L_vec_small[i, ALL_DATA_Y_L_small[i]] = 1
ALL_DATA_Y_L_vec_small = torch.from_numpy(ALL_DATA_Y_L_vec_small).type(torch.FloatTensor)


index_test = [i for i in range(len(ALL_DATA_X_no_L_small))]
random.shuffle(index_train)
ALL_DATA_X_no_L_small = ALL_DATA_X_no_L_small[index_test,:,:]
ALL_DATA_X_no_L_small = ALL_DATA_X_no_L_small.reshape(ALL_DATA_X_no_L_small.shape[0],1,ALL_DATA_X_no_L_small.shape[1],ALL_DATA_X_no_L_small.shape[2])/ 255.
ALL_DATA_X_no_L_small = torch.from_numpy(ALL_DATA_X_no_L_small).type(torch.FloatTensor)

ALL_DATA_Y_no_L_small = ALL_DATA_Y_no_L_small[index_test]  
ALL_DATA_Y_no_L_vec_small = np.zeros((len(ALL_DATA_Y_no_L_small), num_label), dtype=np.float)
for i, label in enumerate(ALL_DATA_Y_no_L_small):
    ALL_DATA_Y_no_L_vec_small[i, ALL_DATA_Y_no_L_small[i]] = 1
ALL_DATA_Y_no_L_vec_small = torch.from_numpy(ALL_DATA_Y_no_L_vec_small).type(torch.FloatTensor)

#####################=========================================================
patch_size=64
ALL_DATA_X_L_larg, ALL_DATA_Y_L_larg, ALL_DATA_X_no_L_larg,ALL_DATA_Y_no_L_larg=Read_img_Gan(h_ipt,w_ipt,n_segments,patch_size,num_label,rownum,colnum,iput_img_original,gt_original,train_patch_ind)


ALL_DATA_X_L_larg = ALL_DATA_X_L_larg[index_train,:,:]
ALL_DATA_X_L_larg = ALL_DATA_X_L_larg.reshape(ALL_DATA_X_L_larg.shape[0],1,ALL_DATA_X_L_larg.shape[1],ALL_DATA_X_L_larg.shape[2])/ 255.
ALL_DATA_X_L_larg = torch.from_numpy(ALL_DATA_X_L_larg).type(torch.FloatTensor)

ALL_DATA_X_no_L_larg = ALL_DATA_X_no_L_larg[index_test,:,:]
ALL_DATA_X_no_L_larg = ALL_DATA_X_no_L_larg.reshape(ALL_DATA_X_no_L_larg.shape[0],1,ALL_DATA_X_no_L_larg.shape[1],ALL_DATA_X_no_L_larg.shape[2])/ 255.
ALL_DATA_X_no_L_larg = torch.from_numpy(ALL_DATA_X_no_L_larg).type(torch.FloatTensor)

############=============================================================
#load the weights from GANs, then train and test the CNN
class Config:
    epoch=int(30)
    batch_size=int(330)
    save_dir='models'
    result_dir='results'
    log_dir='logs'
    lrG=float(0.0002)
    lrD=float(0.0002)
    beta1=float(0.5)
    beta2 =float(0.999)
    gpu_mode=True
    
args=Config()

print("finished")
args.epoch=int(10)

args.model_dir="DCGAN_1G_Class_MSTAR"
run_time=1

from Hierarchy_Semi_DCGAN_FangChengGang import Semi_DCGAN_FangChengGang
#======================================
test_acc_2g_Class=[]
avg_test_acc_2g_Class=0
std_test_acc_2g_Class=0
acc_all_time_2G=[]

for time_i in range (run_time):
    print(" [*] load the DCGAN_1G_Class_MSTAR GANs " + str(time_i))
    args.gan_type= 'DCGAN_1G_Class_MSTAR'
    cnn_2g_Class=Semi_DCGAN_FangChengGang(args,ALL_DATA_X_L_small,ALL_DATA_Y_L_vec_small,ALL_DATA_X_no_L_small,ALL_DATA_Y_no_L_small,ALL_DATA_X_L_larg, ALL_DATA_X_no_L_larg)
    acc_final,acc_all_epoch=cnn_2g_Class.train()
    test_acc_2g_Class.append(acc_final)
    acc_all_time_2G.append(acc_all_epoch)


avg_test_acc_2g_Class = np.array(test_acc_2g_Class).mean()
std_test_acc_2g_Class = np.std(test_acc_2g_Class)
avg_acc_all_time=np.mean(acc_all_time_2G,0)

with open('acc_all_'+'_DCGAN_1G_Class_MSTAR.txt','w') as f:
    f.write(str(avg_acc_all_time))   
    f.write(str(avg_test_acc_2g_Class)) 
    f.write(str(std_test_acc_2g_Class)) 


# ## save the parameter of hierarchical cgans 

torch.save(cnn_2g_Class.D.state_dict(),'./Discriminator_state_dict_6trainImg_Hiera_16_64_Epoch10.pt')


##############################################################################################################


import time
start = time.time()

# # step 3：train CRF using CRF
# ## 利用 微调后的D网络的参数，搭建特征提取网络 ExtractFea
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
from pylab import plt
import torch.utils.data as data
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import sklearn
from skimage.segmentation import slic,mark_boundaries
from skimage import io
from skimage import data,color,morphology,measure
import utils_GAN
class Config:
    epoch=int(30)
    batch_size=int(330)
    save_dir='models'
    result_dir='results'
    log_dir='logs'
    lrG=float(0.0002)
    lrD=float(0.0002)
    beta1=float(0.5)
    beta2 =float(0.999)
    gpu_mode=True
    
args=Config()

class new_discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    
    def __init__(self,args):
        super(new_discriminator, self).__init__()
        self.input_height_small = 16
        self.input_width_small = 16
        self.input_dim = 1
        self.output_dim = 1
        self.class_num = 5
        self.input_height_large = 64
        self.input_width_large = 64
        
        self.conv_small = nn.Sequential(
            
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

        )
        self.fc_small = nn.Sequential(
            nn.Linear(128 * (self.input_height_small // 8) * (self.input_width_small // 8), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            #nn.Linear(1024,160),
            #nn.LeakyReLU(0.2),
  
        )

         
        self.conv_large = nn.Sequential(
            
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

        )
        self.fc_large= nn.Sequential(
            nn.Linear(128 * (self.input_height_large // 8) * (self.input_width_large // 8), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            #nn.Linear(1024,160),
            #nn.LeakyReLU(0.2),
  
        )

        
        self.cl_new = nn.Sequential(
            nn.Linear(1024*2, self.class_num),
            #nn.LogSoftmax() 
            #nn.Sigmoid(),
        )        
        utils_GAN.initialize_weights(self)  


    def forward(self, input1,input2):
        x_large = self.conv_large(input1)
        x_large = x_large.view(-1, 128 * (self.input_height_large // 8) * (self.input_width_large // 8))
        x_large = self.fc_large(x_large)
        
        x_small = self.conv_small(input2)
        x_small = x_small.view(-1, 128 * (self.input_height_small // 8) * (self.input_width_small // 8))
        x_small = self.fc_small(x_small) 
        
        combined = torch.cat((x_large.view(x_large.size(0), -1),
                          x_small.view(x_small.size(0), -1)), dim=1)
        
        c = self.cl_new(combined)
        
        return combined


ExtractFea=new_discriminator(args)


# In[14]:


ALL_pretrained_dict=torch.load('./Discriminator_state_dict_6trainImg_Hiera_16_64_Epoch10.pt')
model_dict=ExtractFea.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in ALL_pretrained_dict.items() if k in model_dict }
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
ExtractFea.load_state_dict(model_dict)
ExtractFea.cuda()


# In[15]:


gan_w=ExtractFea.state_dict()['cl_new.0.weight'].cpu().numpy()
gan_b=ExtractFea.state_dict()['cl_new.0.bias'].cpu().numpy()


# ## 提取每个超像素的特征

def ExtractFeature(ExtractFea, test_loader_large,test_loader_small):
    ExtractFea.eval()
    ALL_Fea=[]
    
    for ii, data_la in enumerate(test_loader_large):
        inputs_l = Variable(torch.from_numpy(test_loader_large[ii].reshape(1,1,64,64).astype("float32")/255.).type(torch.FloatTensor)).cuda()
        inputs_s = Variable(torch.from_numpy(test_loader_small[ii].reshape(1,1,16,16).astype("float32")/255.).type(torch.FloatTensor)).cuda()
        output=ExtractFea(inputs_l,inputs_s)[0]
        ALL_Fea.append(np.array(output.data))
    ALL_Fea=np.array(ALL_Fea) 
    return ALL_Fea


# ### for Graph-CRF

# crop the original input to a set of smaller size images
from torchtools import train_CNN,testCNN,get_dataloader,evaluateCNN,PredictCNN


iput_img_original=image.imread('../../datasets/fangchenggang/A2_广西防城港.jpg')
gt_original=image.imread('../../datasets/fangchenggang/ground_truth2.jpg')

h_ipt=gt_original.shape[0]
w_ipt=gt_original.shape[1]
patch_size_l=64
patch_size_s=16
num_label=5
n_segments=8000
num_train=18
rownum=6
colnum=6
train_patch_ind=[5,10,15,20,21,25,30]


rownum=6
colnum=6
rowheight = h_ipt // rownum
colwidth = w_ipt // colnum
small_input=np.zeros([rownum,colnum,rowheight,colwidth])
small_gt=   np.zeros([rownum,colnum,rowheight,colwidth,3])
#定义一个函数，按照super-pixel的质心，把super pixel截取下来，放到一个patch里面。
#输入为superpixel的质心，patch的size,和一个padding之后的输入图像

def crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s):
    patch_out_l=new_input_padding_l[wi_ipt:wi_ipt+patch_size_l,hi_ipt:hi_ipt+patch_size_l]
    patch_out_s=new_input_padding_s[wi_ipt:wi_ipt+patch_size_s,hi_ipt:hi_ipt+patch_size_s]
    return patch_out_l,patch_out_s

ALL_DATA_X=[]
ALL_DATA_Y=[]

for r in range(rownum):
    for c in range(colnum):#
        iput_img=small_input[r,c,:,:] = iput_img_original[r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth];
        gt      =small_gt[r,c,:,:,:]  = gt_original      [r * rowheight:(r + 1) * rowheight,c * colwidth:(c + 1) * colwidth,:];
        
        segments = slic(iput_img, n_segments=n_segments, compactness=0.5)
        out=mark_boundaries(gt,segments)
        
        from segraph import create_graph
        # Create graph of superpixels 
        vertices, edges = create_graph(segments)
       
        
        #define the patch set all_data_x and its labels
        a_data_x_l=np.zeros((len(vertices),1,patch_size_l,patch_size_l))
        a_data_x_s=np.zeros((len(vertices),1,patch_size_s,patch_size_s))
        a_data_y=[]#np.zeros(len(vertices))
        
        print(len(vertices))
        one_img_fea=[]
        #得到每个super pixel的质心
        segments_label=segments+1  #这里+1，是因为regionprops函数的输入要求是label之后的图像，而label的图像的区域编号是从1开始的
        region_fea=measure.regionprops(segments_label)

          #定义一个边界扩大的patch_size的空矩阵，主要是为了当super pixel位于图像边缘时,把这个iput_img放到new_input_padding中    
        new_input_padding_l=np.zeros([rowheight+patch_size_l,colwidth+patch_size_l])
        new_input_padding_l[patch_size_l/2:-patch_size_l/2,patch_size_l/2:-patch_size_l/2]=iput_img

        new_input_padding_s=np.zeros([rowheight+patch_size_s,colwidth+patch_size_s])
        new_input_padding_s[patch_size_s/2:-patch_size_s/2,patch_size_s/2:-patch_size_s/2]=iput_img

        
        #对所有的super pixel开始循环
        for ind_pixel in range (segments_label.max()):
            
            #计算当前superpixel的质心，为了生成切片，切片以这个质心为中心
            centriod=np.array(region_fea[ind_pixel].centroid).astype("int32")
            wi_ipt=centriod[0]
            hi_ipt=centriod[1]

            #得到这个超像素的所有像素的坐标，根据坐标能够知道这个超像素在GT图中的所有像素值all_pixels
            #根据所有的像素，得到哪一个像素值最多，例如【0,0,0】最多，那这个超像素的标签就是“河流”

            all_pixels=gt[region_fea[ind_pixel].coords[:,0],region_fea[ind_pixel].coords[:,1]]
            n0 = np.bincount(all_pixels[:,0])
            n1 = np.bincount(all_pixels[:,1])  
            n2 = np.bincount(all_pixels[:,2])  
            gt_of_superp=[n0.argmax(),n1.argmax(),n2.argmax()] #gt_of_superp这个超像素中出现最多次的像素值


            if gt_of_superp[0]<=20 and gt_of_superp[1]>=240 and gt_of_superp[2]<=20:  
                a_patch_l,a_patch_s= crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s)
                a_data_x_l[ind_pixel,0,:,:]=a_patch_l
                a_data_x_s[ind_pixel,0,:,:]=a_patch_s
                a_data_y.append(0)
                
      
           # black ---river 
            elif gt_of_superp[0]<=50 and gt_of_superp[1]<=50 and gt_of_superp[2]<=50: 
                a_patch_l,a_patch_s= crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s)
                a_data_x_l[ind_pixel,0,:,:]=a_patch_l
                a_data_x_s[ind_pixel,0,:,:]=a_patch_s
                a_data_y.append(1)

            # red ---urban area 
            elif gt_of_superp[0]>=200 and gt_of_superp[1]<=50 and gt_of_superp[2]<=50: 
                a_patch_l,a_patch_s= crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s)
                a_data_x_l[ind_pixel,0,:,:]=a_patch_l
                a_data_x_s[ind_pixel,0,:,:]=a_patch_s
                a_data_y.append(2)

            # yellow --- framland 
            elif gt_of_superp[0]>=200 and gt_of_superp[1]>=200 and gt_of_superp[2]<=50: 
                a_patch_l,a_patch_s= crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s)
                a_data_x_l[ind_pixel,0,:,:]=a_patch_l
                a_data_x_s[ind_pixel,0,:,:]=a_patch_s
                a_data_y.append(3)
            # blue ---road
            elif gt_of_superp[0]<=50 and gt_of_superp[1]<=50 and gt_of_superp[2]>=200:  
                a_patch_l,a_patch_s= crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s)
                a_data_x_l[ind_pixel,0,:,:]=a_patch_l
                a_data_x_s[ind_pixel,0,:,:]=a_patch_s
                a_data_y.append(2)
            # white ---background
            elif gt_of_superp[0]>=200 and gt_of_superp[1]>=200 and gt_of_superp[2]>=200:  
                a_patch_l,a_patch_s= crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s)
                a_data_x_l[ind_pixel,0,:,:]=a_patch_l
                a_data_x_s[ind_pixel,0,:,:]=a_patch_s
                a_data_y.append(4)

            # other color regions are regarded as the background
            else:
                a_patch_l,a_patch_s= crop_fun(new_input_padding_l,new_input_padding_s,wi_ipt,hi_ipt,patch_size_l,patch_size_s)
                a_data_x_l[ind_pixel,0,:,:]=a_patch_l
                a_data_x_s[ind_pixel,0,:,:]=a_patch_s
                a_data_y.append(4)
        
       
        
        #a_data_loader_l = get_dataloader(a_data_x_l.astype("float32")/255.,a_data_y,batch_size=1, shuffle=False)
        #a_data_loader_s = get_dataloader(a_data_x_s.astype("float32")/255.,a_data_y,batch_size=1, shuffle=False)
        a_data_Fea=ExtractFeature(ExtractFea, a_data_x_l,a_data_x_s)
        print(a_data_Fea.shape)
        
      # define edge feature= x,y,feature
        edge_feature=[]
        for v_edge in edges:
            edge_fea1=a_data_Fea[v_edge[0]]
            edge_fea2=a_data_Fea[v_edge[1]]
            edge_feature.append(np.mean(np.abs(np.array(edge_fea1)-np.array(edge_fea2))))
#         print(np.array(edge_feature).shape)

        one_img_fea.append(a_data_Fea)
        one_img_fea.append(np.array(edges))
        one_img_fea.append(np.array(edge_feature).reshape(-1,1))   
        
        ALL_DATA_X.append(one_img_fea)
        ALL_DATA_Y.append(np.array(a_data_y))

            
ALL_DATA_X=np.array(ALL_DATA_X)
ALL_DATA_Y=np.array(ALL_DATA_Y)



np.save("ALL_DATA_X_EdgeCRF.npy",ALL_DATA_X)
np.save("ALL_DATA_Y_EdgeCRF.npy",ALL_DATA_Y)


ALL_DATA_X=np.load("ALL_DATA_X_EdgeCRF.npy")
ALL_DATA_Y=np.load("ALL_DATA_Y_EdgeCRF.npy")


# ## 数据分成 train 和test

# In[20]:


X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
#train_patch_ind=[10,11,22,23,32,33,44,45, 54,55,66,67, 76,77,88,89,  78,79,90,91, 98,99,110,111, 120,121,132,133]

#train_patch_ind=[5,10,15,20,21,25,30]
train_patch_ind=[0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,22,23,24,26,27,28,29,31,32,33,34,35]
test_patch_ind=range(rownum*colnum)
for ind_patch in train_patch_ind:
    X_train.append(ALL_DATA_X[ind_patch])
    Y_train.append(ALL_DATA_Y[ind_patch])
    test_patch_ind.remove(ind_patch)
    
for ind_patch in test_patch_ind:
    X_test.append(ALL_DATA_X[ind_patch])
    Y_test.append(ALL_DATA_Y[ind_patch])
    
X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)


end = time.time()
print str(end-start)

########################################################################
import time
start = time.time()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score
from pystruct.learners import OneSlackSSVM
from pystruct.datasets import load_snakes
from pystruct.utils import make_grid_edges, edge_list_to_features
from pystruct.models import EdgeFeatureGraphCRF
inference = 'ad3'
# first, train on X with directions only:
crf = EdgeFeatureGraphCRF(inference_method=inference,class_weight=np.array([1.0,1,1,1,1]))#class_weight=np.array([0.15,0.1,0.05,0.05,0.65]
ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, max_iter=200,n_jobs=1,show_loss_every=0)
ssvm.fit(X_train, Y_train)


end = time.time()
print str(end-start)

####################################################################
import time
start = time.time()

from sklearn import metrics
print("start test")
y_pred = ssvm.predict(X_test)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(np.hstack(Y_test), np.hstack(y_pred)))
#np.save("Hierarical_CGAN_CRF_y.npy",y_pred)

target_names = ['no_image','river','urban area', 'framland','background'] #
from sklearn.metrics import classification_report
print(classification_report(np.hstack(Y_test), np.hstack(y_pred), target_names=target_names,digits=4))


Y_test=np.hstack(Y_test)
y_pred=np.hstack(y_pred)
new_Y_test=[]
new_Y_pred=[]
for ii,value in  enumerate( Y_test):
    if value!=0:
        #print(value,y_pred[ii])
        new_Y_test.append(value)
        new_Y_pred.append(y_pred[ii])
    else:
        pass
    
from sklearn.metrics import classification_report
print(classification_report(np.hstack(new_Y_test), np.hstack(new_Y_pred), target_names=target_names,digits=4))

end = time.time()
print str(end-start)
