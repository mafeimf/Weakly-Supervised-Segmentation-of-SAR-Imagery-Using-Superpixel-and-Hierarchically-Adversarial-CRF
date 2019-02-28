import utils_GAN, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from os import listdir
from os.path import isfile,join
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
from visdom import Visdom
from focalloss2d import FocalLoss2d       
from sklearn.metrics import confusion_matrix
class new_discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    
    def __init__(self,args):
        super(new_discriminator, self).__init__()
        self.input_height = 16
        self.input_width = 16
        self.input_dim = 1
        self.output_dim = 1
        self.class_num = 5
        
        self.conv = nn.Sequential(
            
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 8) * (self.input_width // 8), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            #nn.Linear(1024,160),
            #nn.LeakyReLU(0.2),
  
        )
        self.cl_new = nn.Sequential(
            nn.Linear(1024, self.class_num,bias=False),
            #nn.LogSoftmax()
#             nn.Sigmoid(),
        )
        
        utils_GAN.initialize_weights(self)  


    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_height // 8) * (self.input_width // 8))
        x = self.fc(x)
        c = self.cl_new(x)
        return c

class Semi_DCGAN_FangChengGang(object):
    def __init__(self, args,ALL_DATA_X_L, ALL_DATA_Y_L_vec, ALL_DATA_X_no_L, ALL_DATA_Y_no_L_vec):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 1
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.lambda_ = 0.25
        self.model_dir=args.model_dir
        # networks init
        
        self.D = new_discriminator(args)
        
        ALL_pretrained_dict=torch.load(os.path.join(self.save_dir, self.model_dir, 'DCGAN_1G_Class_MSTAR' + '_D_6trainImg_5class.pkl'))
        model_dict=self.D.state_dict()

        pretrained_dict = {k: v for k, v in ALL_pretrained_dict.items() if k in model_dict }
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.D.load_state_dict(model_dict)
        
                                       
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
        
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1,1,1,1])).type(torch.FloatTensor)).cuda()
            #self.CE_loss = FocalLoss2d().cuda()
        else:
            self.BCE_loss = nn.BCELoss() 
            self.CE_loss = nn.CrossEntropyLoss()
            #self.CE_loss = FocalLoss2d()
        print('---------- Networks architecture -------------')
        
        utils_GAN.print_network(self.D)
        print('-----------------------------------------------')

        # load dataset
 
        self.data_X, self.data_Y, self.testdata_X,  self.testdata_Y= ALL_DATA_X_L, ALL_DATA_Y_L_vec, ALL_DATA_X_no_L, ALL_DATA_Y_no_L_vec


    def train(self):
        print("5 classes")
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print('training start!!')
        all_acc=[] 
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            
            for iter in range(len(self.data_X) // self.batch_size):
                
                x_ = self.data_X[iter*self.batch_size:(iter+1)*self.batch_size]
                y_vec_ = self.data_Y[iter*self.batch_size:(iter+1)*self.batch_size]

                if self.gpu_mode:
                    x_,y_vec_ = Variable(x_.cuda()), Variable(y_vec_.cuda())
                else:
                    x_,y_vec_ = Variable(x_), Variable(y_vec_)

                # update D network
                self.D_optimizer.zero_grad()
                C_real = self.D(x_)
                self.D.train()
                #print(torch.max(C_real, 1)[1])
                #print(C_real)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])
                

             
                D_loss = C_real_loss #+C_fake_loss
                self.train_hist['D_loss'].append(D_loss.data[0])
                D_loss.backward()
                self.D_optimizer.step()

            accuracy=0
            
            if((epoch+1)%10==0):  
                self.D.eval()
                all_pred_y=[]
                for iter_test  in range(len(self.testdata_Y) // self.batch_size):

                    test_X = self.testdata_X[iter_test*self.batch_size:(iter_test+1)*self.batch_size]
                    test_Y = self.testdata_Y[iter_test*self.batch_size:(iter_test+1)*self.batch_size]

                    test_X,test_Y = Variable(test_X.cuda()), test_Y
                    test_output = self.D(test_X)  
                    pred_y= torch.max(test_output, 1)[1].data.squeeze() 
                    all_pred_y.append(pred_y.cpu().numpy()) 
                    accuracy = np.float(sum(pred_y.cpu().numpy() == test_Y)) / len(test_Y)+accuracy
                    all_acc.append(accuracy/(iter_test+1))
                all_pred_y=np.array(all_pred_y).squeeze() 
                all_pred_y=all_pred_y.reshape(-1,1)
                print(all_pred_y.shape)
                
                matrix_confu=confusion_matrix(self.testdata_Y[0:all_pred_y.shape[0]],all_pred_y)
                print('Semi_DCGAN_2G_Class_mstar Epoch:', epoch, '|train loss:%.4f'%D_loss, '|test accuracy:%.4f'%(accuracy/(iter_test+1)))  
                print(matrix_confu)    
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)           
#           self.visualize_results((epoch+1))
                 
        
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        
        
        
        return accuracy/(iter_test+1),all_acc#,matrix_confu

           
  
    def visualize_results(self, epoch, fix=True):

        #save feature images
        features = self.D(Variable(self.testdata_X.cuda()))
        samples = features.cpu().data.squeeze()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca3 = PCA(n_components=3)
        x_pca = pca.fit_transform(samples)
        x_pca3 = pca3.fit_transform(samples)
        import visdom

        viz = visdom.Visdom()
#         print(x_pca)
        viz.scatter(X=x_pca[0:600], Y=self.testdata_Y[0:600]+1,opts=dict(colormap='Jet',markersize=12,title="PM"))