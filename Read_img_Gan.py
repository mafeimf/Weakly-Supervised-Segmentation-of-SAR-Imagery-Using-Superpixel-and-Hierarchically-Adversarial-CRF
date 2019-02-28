# coding: utf-8
import numpy as np
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
# 为了Fine-tune 网络Hierarchy CGAN，定义的读图像的数据，能够根据输入的参数 patch_size，得到对应大小的patch
def Read_img_Gan(h_ipt,w_ipt,n_segments,patch_size,num_label,rownum,colnum,iput_img_original,gt_original,train_patch_ind):
    
    def crop_fun(new_input_padding,wi_ipt,hi_ipt,patch_size):
        patch_out=new_input_padding[wi_ipt:wi_ipt+patch_size,hi_ipt:hi_ipt+patch_size]
        return patch_out
    
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
