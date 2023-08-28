import torch
from torch import nn
import torch.nn.functional as F



class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7], use_mask=False):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        G = 1 if use_mask else 0
        group_size = dim_xl // 2
        print(dim_xh, dim_xl,group_size, G)
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+G, data_format='channels_first'),
            nn.Conv2d(group_size + G, group_size + G, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2, 
                      dilation=d_list[0], groups=group_size + G)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+G, data_format='channels_first'),
            nn.Conv2d(group_size + G, group_size + G, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2, 
                      dilation=d_list[1], groups=group_size + G)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+G, data_format='channels_first'),
            nn.Conv2d(group_size + G, group_size + G, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2, 
                      dilation=d_list[2], groups=group_size + G)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+G, data_format='channels_first'),
            nn.Conv2d(group_size + G, group_size + G, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2, 
                      dilation=d_list[3], groups=group_size + G)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + G * 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + G * 4, dim_xl, 1),
            ResConvInRELUBlock(dim_xl, 4, 3)
        )
    def forward(self, xh, xl, mask=None):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1)) if mask is not None else self.g0(torch.cat((xh[0], xl[0]), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1)) if mask is not None else self.g1(torch.cat((xh[1], xl[1]), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1)) if mask is not None else self.g2(torch.cat((xh[2], xl[2]), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1)) if mask is not None else self.g3(torch.cat((xh[3], xl[3]), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        
        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2
        
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        #----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * (1 + self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True)))
        #----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * (1 + self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0))
        x2 = x2.permute(0, 2, 3, 1)
        #----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * (1 + self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0))
        x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class ConvInRELU(nn.Module):
    """ classic combination: conv + batch normalization [+ relu]
        post-activation mode """

    def __init__(self, in_channels, out_channels, ksize, padding, do_act=True, bias=True):
        super(ConvInRELU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.do_act = do_act
        if do_act:
            self.act = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        out = self.norm(self.conv(input))
        if self.do_act:
            out = self.act(out)
        return out


class BottConvInRELU(nn.Module):
    """Bottle neck structure"""

    def __init__(self, channels, ratio, do_act=True, bias=True):
        super(BottConvInRELU, self).__init__()
        self.conv1 = ConvInRELU(channels, channels//ratio, ksize=1, padding=0, do_act=True, bias=bias)
        self.conv2 = ConvInRELU(channels//ratio, channels//ratio, ksize=3, padding=1, do_act=True, bias=bias)
        self.conv3 = ConvInRELU(channels//ratio, channels, ksize=1, padding=0, do_act=do_act, bias=bias)

    def forward(self, input):
        out = self.conv3(self.conv2(self.conv1(input)))
        return out


class ResConvInRELUBlock(nn.Module):
    """ block with bottle neck conv"""

    def __init__(self, channels, ratio, num_convs):
        super(ResConvInRELUBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            if i != num_convs - 1:
                layers.append(BottConvInRELU(channels, ratio, True))
            else:
                layers.append(BottConvInRELU(channels, ratio, False))

        self.ops = nn.Sequential(*layers)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        output = self.ops(input)
        return self.act(input + output)



class EGEGenerator(nn.Module):
    def __init__(self, output_channels=1, input_channels=1, c_list=[16,32,64,128,256,512], bridge=True):
        super(EGEGenerator, self).__init__()
        self.bridge = bridge
        c_list = [c * 2 for c in c_list]

        self.ebn1 = nn.InstanceNorm2d(c_list[0])
        self.ebn2 = nn.InstanceNorm2d(c_list[1])
        self.ebn3 = nn.InstanceNorm2d(c_list[2])
        self.ebn4 = nn.InstanceNorm2d(c_list[3])
        self.ebn5 = nn.InstanceNorm2d(c_list[4])
        self.dbn1 = nn.InstanceNorm2d(c_list[4])
        self.dbn2 = nn.InstanceNorm2d(c_list[3])
        self.dbn3 = nn.InstanceNorm2d(c_list[2])
        self.dbn4 = nn.InstanceNorm2d(c_list[1])
        self.dbn5 = nn.InstanceNorm2d(c_list[0])

        #self.dbn_final = nn.InstanceNorm2d(c_list[0])

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 2, stride=2, padding=0),
            self.ebn1,
            nn.LeakyReLU(0.2, True),
            BottConvInRELU(c_list[0], 2)
        )

        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 2, stride=2, padding=0),
            self.ebn2,
            nn.LeakyReLU(0.2, True),
            ResConvInRELUBlock(c_list[1], 4, 2)
        ) 

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 2, stride=2, padding=0),
            self.ebn3,
            nn.LeakyReLU(0.2, True),
            ResConvInRELUBlock(c_list[2], 4, 2)
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 2, stride=2, padding=0),
            self.ebn4, 
            nn.LeakyReLU(0.2, True),
            ResConvInRELUBlock(c_list[3], 4, 3)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 2, stride=2, padding=0),
            self.ebn5, 
            nn.LeakyReLU(0.2, True),
            ResConvInRELUBlock(c_list[4], 4, 3)
        )
        self.encoder6 = nn.Sequential(
            ConvInRELU(c_list[4], c_list[5], 3, 1, 1),
            BottConvInRELU(c_list[5], 4, )
        )

        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')


        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[5]),
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1),
            self.dbn1,
            ResConvInRELUBlock(c_list[4], 4, 3)
        ) 
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[4]),
            nn.Conv2d(c_list[4], c_list[4], 3, stride=1, padding=1),
            nn.ConvTranspose2d(c_list[4], c_list[3],kernel_size=4, stride=2, padding=1, bias=True),
            self.dbn2,
            ResConvInRELUBlock(c_list[3], 4, 3)
        ) 
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[3]),
            nn.Conv2d(c_list[3], c_list[3], 3, stride=1, padding=1),
            nn.ConvTranspose2d(c_list[3], c_list[2],kernel_size=4, stride=2, padding=1, bias=True),
            self.dbn3,
            ResConvInRELUBlock(c_list[2], 4, 3)
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[2], 3, stride=1, padding=1),
            nn.ConvTranspose2d(c_list[2], c_list[1],kernel_size=4, stride=2, padding=1, bias=True),
            self.dbn4,
            ResConvInRELUBlock(c_list[1], 4, 3)
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[1], 3, stride=1, padding=1),
            nn.ConvTranspose2d(c_list[1], c_list[0],kernel_size=4, stride=2, padding=1, bias=True),
            self.dbn5,
            ResConvInRELUBlock(c_list[0], 2, 3)
        )  

        self.final = nn.Sequential(
            nn.Conv2d(c_list[0] * 3, c_list[0] * 3, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(c_list[0] * 3, c_list[0],kernel_size=4, stride=2, padding=1, bias=True),
            ConvInRELU(c_list[0], c_list[0], ksize=3, padding=1, do_act=True),
            nn.Conv2d(c_list[0], c_list[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(c_list[0], output_channels, kernel_size=1,bias=False),
            nn.Tanh()
        )
        


    def forward(self, x):
        out = self.encoder1(x)
        t1 = out # b, c0, H/2, W/2
        
        out = self.encoder2(out)
        t2 = out # b, c1, H/4, W/4 
        
        out = self.encoder3(out)
        t3 = out # b, c2, H/8, W/8 
        
        out = self.encoder4(out)
        t4 = out # b, c3, H/16, W/16 
        
        out = self.encoder5(out)
        t5 = out # b, c4, H/32, W/32 
        
        out = self.encoder6(out) # b, c5, H/32, W/32
        t6 = out# b, c5, H/32, W/32
        

        out5 = self.decoder1(out) # b, c4, H/32, W/32
        t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5)
        
        out4 = self.decoder2(out5) # b, c3, H/16, W/16
        t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16

        out3 = self.decoder3(out4) # b, c2, H/8, W/8
        t3 = self.GAB3(t4, t3)
        #out3 = torch.concat([out3, t3], dim=1) # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)

        out2 = self.decoder4(out3) # b, c1, H/4, W/4
        t2 = self.GAB2(t3, t2)
        #out2 = torch.concat([out2, t2], dim=1) # b, c1, H/4, W/4 
        out2 = torch.add(out2, t2)

        out1 = self.decoder5(out2) # b, c0, H/2, W/2
        t1_ = self.GAB1(t2, t1)
        out1 = torch.concat([out1, t1, t1_], dim=1) # b, c0 * 2 + c1, H/2, W/2

        out0 = self.final(out1)
        return out0