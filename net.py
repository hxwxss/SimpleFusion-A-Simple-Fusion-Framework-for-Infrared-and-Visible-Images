import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import pdb
           


class L_net(nn.Module):
    def __init__(self,in_channel, num):
        super(L_net, self).__init__()
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, 8, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 32, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 8, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, in_channel, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_net(input)+input)


class R_net(nn.Module):
    def __init__(self,in_channel, num):
        super(R_net, self).__init__()

        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, 8, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 32, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 8, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, in_channel, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.R_net(input)+input)

class N_net(nn.Module):
    def __init__(self,in_channel,num):
        super(N_net, self).__init__()
        self.N_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, 8, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(8,16, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, 1, 0),            
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 8, 3, 1, 0),
            nn.ReLU(),     
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, in_channel, 3, 1, 0),
        )

    def forward(self, input):
        # return torch.sigmoid( 0.8*self.N_net(input) + 0.2 * input)
        return torch.sigmoid(self.N_net(input)+input)


class decomposer(nn.Module):

    def __init__(self,in_channel,num):
        super(decomposer, self).__init__()        
        self.L_net = L_net(in_channel,num)
        self.R_net = R_net(in_channel,num)
        self.N_net = N_net(in_channel,num)        

    def forward(self, input):
        x = self.N_net(input)
        L = self.L_net(x)
        R = self.R_net(x)
        # pdb.set_trace()
        return L, R, x

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# class Decoder_simple(nn.Module):
#     def __init__(self, s, in_channel, stride, fusion_type):
#         super(Decoder_simple, self).__init__()
#         self.type = fusion_type


#         self.conv_ReLX = decoder_ConvLayer(1, in_channel, s, stride)
#         self.conv_ReRX = decoder_ConvLayer(in_channel, in_channel, s, stride)
#         self.conv_ReLY = decoder_ConvLayer(1, in_channel, s, stride)
#         self.conv_ReRy = decoder_ConvLayer(in_channel, in_channel, s, stride)

#         if self.type.__contains__('cat'):
#             # cat
#             self.conv_ReL = decoder_ConvLayer(2 * in_channel, in_channel, s, stride)
#             self.conv_ReH = decoder_ConvLayer(2 * in_channel, in_channel, s, stride)

#             # 2---->4
#             self.conv_ReL1 = decoder_ConvLayer(2 * in_channel, in_channel, s, stride)
#             self.conv_ReH1 = decoder_ConvLayer(2 * in_channel, in_channel, s, stride)

#         else:
#             # add
#             self.conv_ReL = decoder_ConvLayer(in_channel, in_channel, s, stride)
#             self.conv_ReH = decoder_ConvLayer(in_channel, in_channel, s, stride)

#     def forward(self, LX, RX, LY, Ry):
#         # get loww parts and sparse parts
#         x_l = self.conv_ReLX(LX)
#         x_r = self.conv_ReRX(RX)
#         y_l = self.conv_ReLY(LY)
#         y_r = self.conv_ReRy(Ry )

#         # pdb.set_trace()
#         # reconstructure
#         if self.type.__contains__('cat'):
#             # cat
#             low = self.conv_ReL(torch.cat([x_l, y_l], 1))
#             high = self.conv_ReH(torch.cat([x_r, y_r], 1))

#             lh1 =  self.conv_ReL1(torch.cat([x_l, y_r], 1))
#             hl1 =  self.conv_ReH1(torch.cat([x_r, y_l], 1))
#             # pdb.set_trace()
#         else:
#             # add
#             low = self.conv_ReL(x_l + y_l)
#             high = self.conv_ReH(x_r + y_r)

#         out = low +  high 
#         # +  hl1 + lh1

#         return out, x_l, x_r, y_l, y_r, low, high


class great_NET(nn.Module):

    def __init__(self,in_channel,num):
        super(great_NET, self).__init__()

        self.decomposer_X = decomposer(in_channel,num)
        self.decomposer_Y = decomposer(in_channel,num)

    def forward(self, x, y):

        Lx, Rx, Nx = self.decomposer_X(x)
        Ly, Ry, Ny = self.decomposer_Y(y)
        # R = (Rx + Ry)
        R = Rx + Ry
        L = Lx + Ly
        f = L * R

        # f, x_l, x_h, y_l, y_h, fl, fh = self.decoder(Lx, Rx, Ly, Ry)
        # pdb.set_trace()
        # out = { 'Lx': Lx, 'Ly': Ly, 'Rx': Rx, 'Ry': Ry, 'f': f, 'x_l': x_l, 'x_h': x_h, 'y_l': y_l, 'y_h': y_h, 'fl': fl, 'fh': fh, 'Nx': Nx, 'Ny': Ny}

        out = { 'Lx': Lx, 'Ly': Ly, 'Rx': Rx, 'Ry': Ry, 'Nx': Nx, 'Ny': Ny, 'f': f}
        return out
    

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        b, c, h, w = X.shape
        if c == 1:
            X = X.repeat(1, 3, 1, 1)

        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        # [relu1_2, relu2_2, relu3_3, relu4_3]
        return [relu1_2, relu2_2, relu3_3, relu4_3]