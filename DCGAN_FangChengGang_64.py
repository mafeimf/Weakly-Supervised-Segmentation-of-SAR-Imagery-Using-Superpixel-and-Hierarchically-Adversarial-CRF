import utils_GAN, torch, time, os, pickle, utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, dataset = 'mnist'):
        super(generator, self).__init__()

        
        self.input_height_large = 64
        self.input_width_large = 64
        self.input_dim = 62
        self.output_dim = 1                
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * (self.input_height_large // 8) * (self.input_width_large // 8)),
            nn.BatchNorm1d(128 * (self.input_height_large // 8) * (self.input_width_large // 8)),
            nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
#             nn.BatchNorm2d(self.output_dim),
            
        )
        
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height_large // 8), (self.input_width_large // 8))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset = 'mnist'):
        super(discriminator, self).__init__()

        self.input_height_large = 64
        self.input_width_large = 64
        self.input_dim = 1
        self.output_dim = 1  
        self.class_num = 5
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
        self.dc_large = nn.Sequential(     
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl_large = nn.Sequential(
            nn.Linear(1024, self.class_num,bias=False),
            #nn.LogSoftmax()
#             nn.Sigmoid(),
        )

        utils_GAN.initialize_weights(self)



    def forward(self, input):
        x = self.conv_large(input)
        x = x.view(-1, 128 * (self.input_height_large // 8) * (self.input_width_large // 8))
        x = self.fc_large(x)
        d = self.dc_large(x)
        c = self.cl_large(x)

        return  d,c #    

class DCGAN_1G_Class_MSTAR_64(object):
    def __init__(self, args,ALL_DATA_X_L, ALL_DATA_Y_L_vec, ALL_DATA_X_no_L):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 9
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir      
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.lambda_ = 0.25

        # networks init
        self.G0 = generator()
        self.D = discriminator()
        
        self.G0_optimizer = optim.Adam(self.G0.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
         
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G0.cuda()
            
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G0)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # load dataset
        self.data_X = ALL_DATA_X_no_L
       
        self.data_X1 = ALL_DATA_X_L
        self.data_Y1 = ALL_DATA_Y_L_vec

        self.z_dim = 62

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

    def train(self):
        print("5 classes")
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []

        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))

        self.D.train()
        print('training start!!')
        
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            self.G0.train()
            for iter in range(len(self.data_X) // self.batch_size):



                x_ = self.data_X[iter*self.batch_size:(iter+1)*self.batch_size]
                
                x_1 = self.data_X1[iter*(self.batch_size*len(self.data_X1)//len(self.data_X)):(iter+1)*(self.batch_size*len(self.data_X1)//len(self.data_X))]#[iter*self.batch_size:(iter+1)*self.batch_size]
                y_vec_ = self.data_Y1[iter*(self.batch_size*len(self.data_X1)//len(self.data_X)):(iter+1)*(self.batch_size*len(self.data_X1)//len(self.data_X))]               
                z_0 = torch.rand((self.batch_size, self.z_dim))
   
                if self.gpu_mode:
                    x_, z_0,y_vec_,x_1 = Variable(x_.cuda()), Variable(z_0.cuda()),  Variable(y_vec_.cuda()), Variable(x_1.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                # update D network
                self.D_optimizer.zero_grad()

                D_real ,C_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                
                D_real_no ,C_real = self.D(x_1)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])
                
                
                G_0 = self.G0(z_0)

                
                D_fake0,C_fake = self.D(G_0)
                D_fake_loss0 = self.BCE_loss(D_fake0, self.y_fake_)
    
                D_loss = D_real_loss + D_fake_loss0+C_real_loss#+ D_fake_loss1+ D_fake_loss2+ D_fake_loss3+ D_fake_loss4
                self.train_hist['D_loss'].append(D_loss.data[0])
                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G0_optimizer.zero_grad()
                   
                G_0 = self.G0(z_0)
      
                D_fake0,C_fake = self.D(G_0)
                
                
                G_loss0 = self.BCE_loss(D_fake0, self.y_real_)
                

                G_loss=G_loss0
                self.train_hist['G_loss'].append(G_loss.data[0])
                
                G_loss.backward()
                self.G0_optimizer.step()


                if ((iter + 1) % 30) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), (len(self.data_X) // self.batch_size), D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)           
#           self.visualize_results((epoch+1))
            
            if ((epoch + 1) % 5) == 0:
                gen_num=0
                self.save(epoch=epoch)
                while(gen_num<1):
                    gen_num = gen_num+1
                    self.visualize_results((epoch+1), fix=False, gen_num=gen_num)
           
   
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!")

        
#         utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name, self.epoch)
        #utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)
        print("saved training results")
        
        
    def visualize_results(self, epoch, fix=True,gen_num=0):
        self.G0.eval()

        if not os.path.exists(self.result_dir + '/' + self.model_name):
            os.makedirs(self.result_dir +  '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G0(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

            samples = self.G0(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)*255
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.result_dir + '/'  + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '_'+str(gen_num)+'.png')

    def save(self,epoch):
        save_dir = os.path.join(self.save_dir, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        #torch.save(self.D.state_dict(),'./Discriminator_state_dict.pt')
        torch.save(self.G0.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_epoch_'+str(epoch)+'_D_6trainImg_5class_64.pkl'))
        #torch.save(self.D.state_dict(),'./Discriminator_state_dict_6trainImg_CNN.pt')

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.G0.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
