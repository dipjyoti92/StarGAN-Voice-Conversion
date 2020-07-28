import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Down2d(nn.Module):

    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.BatchNorm2d(out_channel, affine=True, track_running_stats=True)
        self.glu = nn.GLU(dim=1)
    
    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x1 = self.glu(x1)
        
        return x1
            
        
class Up2d(nn.Module):

    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Up2d, self).__init__()
        self.c1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.BatchNorm2d(out_channel, affine=True, track_running_stats=True)
        self.glu = nn.GLU(dim=1)
         
    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x1 = self.glu(x1)
        
        return x1


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_speakers=70,repeat_num=6):                                                  
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            Down2d(1, 32, (3,9), (1,1), (1,4)),
            Down2d(16, 64, (4,8), (2,2), (1,3)),
            Down2d(32, 128, (4,8), (2,2), (1,3)),
            Down2d(64, 64, (3,5), (1,1), (1,2)),
            Down2d(32, 10, (9,5), (9,1), (1,2))
        )

        
        self.up1 = Up2d(75, 64, (9,5), (9,1), (0,2))
        self.up2 = Up2d(102, 128, (3,5), (1,1), (1,2))
        self.up3 = Up2d(134, 64, (4,8), (2,2), (1,3))
        self.up4 = Up2d(102, 32, (4,8), (2,2), (1,3))

        self.deconv = nn.ConvTranspose2d(86, 1, (3,9), (1,1), (1,4))

    def forward(self, x, c):						# torch.Size([32, 1, 36, 512])

        x = self.downsample(x)						 # torch.Size([32, 5, 1, 128])

        c = c.view(c.size(0), c.size(1), 1, 1)
        
        c1 = c.repeat(1, 1, x.size(2), x.size(3))                        # torch.Size([32, 70, 1, 128])
        x = torch.cat([x, c1], dim=1)                                    # torch.Size([32, 75, 1, 128])
        x = self.up1(x) 						 # torch.Size([32, 64, 9, 128])

        c2 = c.repeat(1,1,x.size(2), x.size(3))				 # torch.Size([32, 70, 9, 128])
        x = torch.cat([x, c2], dim=1)					 # torch.Size([32, 134, 9, 128])
        x = self.up2(x)							 # torch.Size([32, 128, 9, 128])

        c3 = c.repeat(1,1,x.size(2), x.size(3))				 # torch.Size([32, 70, 9, 128])
        x = torch.cat([x, c3], dim=1)					 # torch.Size([32, 198, 9, 128])
        x = self.up3(x)							 # torch.Size([32, 64, 18, 256])

        c4 = c.repeat(1,1,x.size(2), x.size(3)) 			 # torch.Size([32, 70, 18, 256])
        x = torch.cat([x, c4], dim=1)					 # torch.Size([32, 134, 18, 256])
        x = self.up4(x)   						 # torch.Size([32, 32, 36, 512])

        c5 = c.repeat(1,1, x.size(2), x.size(3)) 			 # torch.Size([32, 70, 36, 512])
        x = torch.cat([x, c5], dim=1)					 # torch.Size([32, 102, 36, 512])

        x = self.deconv(x)						 # torch.Size([32, 1, 36, 512])

        return x


class Discriminator(nn.Module):

    def __init__(self, input_size=(36, 512), conv_dim=64, repeat_num=5, num_speakers=70):
        super(Discriminator, self).__init__()
        
        self.d1 = Down2d(71, 32, (3,9), (1,1), (1,4))
        self.d2 = Down2d(86, 32, (3,8), (1,2), (1,3))    
        self.d3 = Down2d(86, 32, (3,8), (1,2), (1,3))    
        self.d4 = Down2d(86, 32, (3,6), (1,2), (1,2)) 
        
        self.conv = nn.Conv2d(86, 1, (36,5), (36,1), (0,2))
        self.pool = nn.AvgPool2d((1,64))
        
    def forward(self, x, c):						# x=torch.Size([32, 1, 36, 256])

        c = c.view(c.size(0), c.size(1), 1, 1)
        c1 = c.repeat(1, 1, x.size(2), x.size(3))			# torch.Size([32, 70, 36, 256])
   
        x = torch.cat([x, c1], dim=1)					# torch.Size([32, 71, 36, 256])
        x = self.d1(x)							# torch.Size([32, 32, 36, 256])

        c2 = c.repeat(1, 1, x.size(2), x.size(3))			# torch.Size([32, 70, 36, 256])
        x = torch.cat([x, c2], dim=1)					# torch.Size([32, 102, 36, 256])
        x = self.d2(x)							# torch.Size([32, 32, 36, 128])

        c3 = c.repeat(1, 1, x.size(2), x.size(3))			# torch.Size([32, 70, 36, 128])
        x = torch.cat([x, c3], dim=1)					# torch.Size([32, 102, 36, 128])
        x = self.d3(x)			 				# torch.Size([32, 32, 36, 128])
       
        c4 = c.repeat(1, 1, x.size(2), x.size(3))			# torch.Size([32, 70, 36, 128])
        x = torch.cat([x, c4], dim=1)					# torch.Size([32, 102, 36, 128])
        x = self.d4(x)							# torch.Size([32, 32, 36, 64])

        c5 = c.repeat(1, 1, x.size(2), x.size(3))			# torch.Size([32, 70, 36, 64])
        x = torch.cat([x, c5], dim=1)					# torch.Size([32, 102, 36, 64])
        x = self.conv(x)						# torch.Size([32, 1, 1, 64])
       
        x = self.pool(x)						# torch.Size([32, 1, 1, 1])
        x = torch.squeeze(x)						# torch.Size([32])
        x = torch.sigmoid(x)						# torch.Size([32])
        return x

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.main = nn.Sequential(
            Down2d(1, 8, (4,4), (2,2), (5,1)),				# torch.Size([32, 8, 22, 256])
            Down2d(4, 16, (4,4), (2,2), (0,1)),				# torch.Size([32, 16, 11, 128])
            Down2d(8, 32, (4,4), (2,2), (0,1)),			        # torch.Size([32, 32, 4, 64])
            Down2d(16, 64, (3,4), (2,2), (0,1)),			# torch.Size([32, 64, 1, 32])
            nn.Conv2d(32, 70, (1,4), (1,2), (0,1)),			# torch.Size([32, 70, 1, 16])
            nn.AvgPool2d((1,16)),					# torch.Size([32, 70, 1, 1])
            nn.Softmax()
        )
        
    def forward(self, x):					# torch.Size([32, 1, 36, 512])			
       x = self.main(x)
       x = x.view(x.size(0), x.size(1))				# torch.Size([32, 70])
       return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_loader('/scratch/sxliu/data_exp/VCTK-Corpus-22.05k/mc/train', 16, 'train', num_workers=1)
    data_iter = iter(train_loader)
    G = Generator().to(device)
    D = Discriminator().to(device)
    for i in range(10):
        mc_real, spk_label_org, acc_label_org, spk_acc_c_org = next(data_iter)
        mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
        mc_real = mc_real.to(device)                         # Input mc.
        spk_label_org = spk_label_org.to(device)             # Original spk labels.
        acc_label_org = acc_label_org.to(device)             # Original acc labels.
        spk_acc_c_org = spk_acc_c_org.to(device)             # Original spk acc conditioning.
        mc_fake = G(mc_real, spk_acc_c_org)
        print(mc_fake.size())
        out_src, out_cls_spks, out_cls_emos = D(mc_fake)



