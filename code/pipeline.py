import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from utils.utils import *
from utils.MSSSIM import BiMSSSIM
from utils.datasets import ImageDataset,ValDataset, TestDataset
from modules.discriminator import *
from modules.generator import *
from modules.registration import Reg
from torchvision.transforms import RandomAffine,ToPILImage
from torchvision import utils as vutils
from modules.transformer import *
#from utils.loss import *
from utils.loss import NCC

from skimage import measure
from modules.SAMencoder import *
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2


class Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.netG_A2B = EGEGenerator(output_channels=config['output_nc'], input_channels=config['input_nc']).cuda()
        self.netG_B2A = EGEGenerator(output_channels=config['input_nc'], input_channels=config['output_nc']).cuda()
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                           lr=config['lr'] * 2 , betas=(0.5, 0.999))
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda = self.rule)

        self.netD_B = PatchDiscriminator(config['input_nc'], n_layers=4).cuda()
        self.netD_A = PatchDiscriminator(config['input_nc'], n_layers=4).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'] , betas=(0.5, 0.999))
        self.scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda = self.rule)
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'] , betas=(0.5, 0.999))
        self.scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda = self.rule)

        if config['SAMpercept'] is True:
            self.sam_encoer, self.sam_transform, _ = regist_sam_image_encoder(config['sam_path'])
            self.sam_encoer = self.sam_encoer.cuda().eval()

        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'] / 2 , betas=(0.5, 0.999))
    
    def rule(self, epoch):
        return 1.0 - max(0, epoch - self.config['decay_start_epoch']) / (self.config['n_epochs'] + 1 - self.config['decay_start_epoch'])

    def save_image_tensor2cv2(self, input_tensor: torch.Tensor, filename):
        assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
        input_tensor = input_tensor.clone().detach()
        input_tensor = input_tensor.to(torch.device('cpu'))
        input_tensor = input_tensor.squeeze(0)
        input_tensor = (input_tensor + 1.) / 2
        input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        cv2.imwrite(filename, input_tensor)
    
    def PSNR(self,fake,real):
       #print(fake.shape, real.shape)
       x,y = np.where(real!= -1)# Exclude background

       mse = np.mean(((fake[x][y]+1)/2. - (real[x][y]+1)/2.) ** 2 )
       if mse < 1.0e-10:
          return 100
       else:
           PIXEL_MAX = 1
           return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
            
    def MAE(self,fake,real):
        #print(fake.shape, real.shape)
        #print(np.where(real!= -1))
        x,y = np.where(real!= -1)  # Exclude background
        mae = np.abs(fake[x,y]-real[x,y]).mean()
        return mae/2     #from (-1,1) normaliz  to (0,1)

            
    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
        tans_x = cv2.normalize(dir_x, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_x[tans_x<=150] = 0
        tans_x = cv2.applyColorMap(tans_x, cv2.COLORMAP_JET)
        tans_y = cv2.normalize(dir_y, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #tans_y[tans_y<=150] = 0
        tans_y = cv2.applyColorMap(tans_y, cv2.COLORMAP_JET)
        gradxy = cv2.addWeighted(tans_x, 0.5,tans_y, 0.5, 0)

        cv2.imwrite(root, gradxy) 

    def inference(self):
        val_transforms = [transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,)),]
        self.test_data = DataLoader(TestDataset(self.config['test_dataroot'], transforms_ =val_transforms),
                                batch_size=1, shuffle=False, num_workers=self.config['n_cpu'])

        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        self.netG_A2B.eval()
        with torch.no_grad():
                for _, batch in enumerate(self.test_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    
                    fake_B = self.netG_A2B(real_A)
                    fake_B[real_A==-1] = -1
                    save_path = os.path.join(self.config['image_save'], batch['case'][0])
                    self.save_image_tensor2cv2(fake_B, save_path)


    def train(self, ):
        level = self.config['noise_level']  # set noise level

        transforms_1 = [transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level],fill=-1),
                        Resize(size_tuple = (self.config['size'], self.config['size']))]


        transforms_2 = [transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02],fill=-1),
                        Resize(size_tuple = (self.config['size'], self.config['size']))]

        self.dataloader = DataLoader(ImageDataset(self.config['dataroot'], level, 
                                     transforms_1=transforms_1, transforms_2=transforms_2, 
                                     unaligned=False, B2A=self.config['BtoA']),
                                     batch_size=self.config['batchSize'], shuffle=True, num_workers=self.config['n_cpu'])

        val_transforms = [transforms.ToTensor(),
                          transforms.Normalize((0.5,), (0.5,)),]
        
        self.val_data = DataLoader(ValDataset(self.config['val_dataroot'], 
                                    transforms_ =val_transforms, unaligned=False, B2A=self.config['BtoA']),
                                    batch_size=1, shuffle=False, num_workers=self.config['n_cpu'])

        Tensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
        self.input_A = Tensor(self.config['batchSize'], self.config['input_nc'], self.config['size'], self.config['size'])
        self.input_B = Tensor(self.config['batchSize'], self.config['output_nc'], self.config['size'], self.config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        self.logger = Logger(self.config['name'],self.config['port'],self.config['n_epochs'], len(self.dataloader)) 
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        #self.ncc = NCC()
        self.ssim = BiMSSSIM(channel=1, window_size=15)

        p = 0.1
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()
                self.optimizer_D_A.zero_grad()
                self.optimizer_D_B.zero_grad()
                # GAN loss
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake).cuda()) 
                
                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.ones_like(pred_fake).cuda())
                
                Trans = self.R_A(fake_B,real_B) 
                SysRegist_A2B = self.spatial_transform(fake_B,Trans)

                SR_loss = (1-p) * self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B,real_B) \
                          + p *self.config['Corr_lamda'] * self.L1_loss(fake_B,real_B)
                SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                #ncc_loss = self.ncc(fake_B, real_B) + self.ncc(fake_A, real_A) / 2

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B)

                # Total loss
                loss_Total = loss_GAN_A2B * 2 + loss_GAN_B2A * 2 + loss_cycle_ABA + loss_cycle_BAB + SR_loss + SM_loss
                if epoch >= 10:
                    ssim_loss = self.ssim(fake_B, real_B)
                    loss_Total += ssim_loss * 0.2

                if self.config['SAMpercept'] is True and epoch >= 50:
                    _realB = tensor_transform(real_B, self.sam_transform)
                    per_realB = self.sam_encoer(_realB)
                    _fakeB = tensor_transform(fake_B, self.sam_transform)
                    per_fakeB = self.sam_encoer(_fakeB)
                    loss_Per_A2B = self.L1_loss(per_realB,per_fakeB)
                    loss_per = loss_Per_A2B * self.config['Per_lambda'] 
                    loss_Total += loss_per
                loss_Total.backward()
                self.optimizer_G.step()
                self.optimizer_R_A.step()
                
                ###### Discriminator A ######
                        
                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real).cuda())
                # Fake loss
                fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake).cuda())

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) 
                loss_D_A.backward()
                self.optimizer_D_A.step()
                # torch.nn.utils.clip_grad_norm_(parameters=self.netD_A.parameters(), max_norm=10, norm_type=2)

                    
                    
                ###################################

                ###### Discriminator B ######

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real).cuda())

                # Fake loss
                fake_B1 = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B1.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, torch.zeros_like(pred_fake).cuda())

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) 
                loss_D_B.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=self.netD_B.parameters(), max_norm=10, norm_type=2)

                #if (i % self.config['d_step_freq'] == 0) or (i <= 50000):
                self.optimizer_D_B.step()
                #self.optimizer_D_B.zero_grad()
                ################################### 

                trans_x = Trans[:, [0]]
                trans_y = Trans[:, [1]]
                trans = torch.abs(trans_x) / 2 + torch.abs(trans_y) / 2
                images={'real_A': real_A, 'real_B': real_B, 'fake_B': fake_B, 'deform':trans, 'fake_B_warped':SysRegist_A2B}
                if epoch >= 50:
                    self.config['Corr_lamda'] = 5
                    self.config['Cyc_lamda'] = 20
                    loss_ = {'loss_D_B': loss_D_B, 'loss_G_adv':loss_GAN_A2B,'SR_loss':SR_loss, 'RegSmotth_loss100x':SM_loss * 100, 'PerLoss':loss_per, 'SSIMLoss':ssim_loss}
                else:
                    loss_ = {'loss_D_B': loss_D_B, 'loss_G_adv':loss_GAN_A2B,'SR_loss':SR_loss, 'RegSmotth_loss100x':SM_loss * 100}
                self.logger.log(loss_, images=images)#,'SR':SysRegist_A2B

            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
            torch.save(self.netG_B2A.state_dict(), self.config['save_root'] + 'netG_B2A.pth')
            torch.save(self.R_A.state_dict(), self.config['save_root'] + 'Regist.pth')
            torch.save(self.netD_A.state_dict(), self.config['save_root'] + 'netD_A.pth')
            torch.save(self.netD_B.state_dict(), self.config['save_root'] + 'netD_B.pth')
            if epoch % 20 == 0:
                torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B_%d.pth'%epoch)
            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()

            #############val###############
            with torch.no_grad():
                MAE = 0
                SSIM = 0
                SSIM1 = 0
                num = 0
                #NCC = 0
                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A']))
                    real_B1 = Variable(self.input_B.copy_(batch['B']))
                    real_B = real_B1.detach().cpu().numpy().squeeze()
                    fake_B1 = self.netG_A2B(real_A)
                    fake_B = fake_B1.detach().cpu().numpy().squeeze()
                    Trans = self.R_A(fake_B1,real_B1) 
                    SysRegist_A2B = self.spatial_transform(fake_B1,Trans).detach().cpu().numpy().squeeze()
                    mae = self.MAE(fake_B,real_B)
                    ssim = compare_ssim(fake_B,real_B)
                    ssim_trans = compare_ssim(SysRegist_A2B,real_B)
                    #ncc_loss = self.ncc(fake_B1, real_B1).detach().cpu().item()
                    #NCC += ncc_loss
                    MAE += mae
                    num += 1
                    SSIM += ssim
                    SSIM1 += ssim_trans

                print ('Val MAE:',MAE/num)
                print ('Val SSIM:',SSIM/num)
                print ('Val SSIM Trans:',SSIM1/num)
                #print ('LNCC:',NCC/num)
            

        


                        




        


        