import torch
from torch import nn
from torch.nn import functional as F


class TPAVIModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot', 
                 dimension=3, bn_layer=True):
        """
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation 
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(TPAVIModule, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        ## add align channel
        self.align_channel = nn.Linear(128, in_channels)
        self.norm_layer=nn.LayerNorm(in_channels)

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )

            
    def forward(self, x, audio=None):
        """
        args:
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
            audio: (N, T, C)
        """

        audio_temp = 0
        batch_size, C = x.size(0), x.size(1)
        if audio is not None:
            # print('==> audio.shape', audio.shape)
            H, W = x.shape[-2], x.shape[-1]
            audio_temp = self.align_channel(audio) # [bs, T, C]
            audio = audio_temp.permute(0, 2, 1) # [bs, C, T]
            audio = audio.unsqueeze(-1).unsqueeze(-1) # [bs, C, T, 1, 1]
            audio = audio.repeat(1, 1, 1, H, W) # [bs, C, T, H, W]
        else:
            audio = x

        # (N, C, THW)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # [bs, C, THW]
        # print('g_x.shape', g_x.shape)
        # g_x = x.view(batch_size, C, -1)  # [bs, C, THW]
        g_x = g_x.permute(0, 2, 1) # [bs, THW, C]

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = audio.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # [bs, C', THW]
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, -1) # [bs, C', THW]
            theta_x = theta_x.permute(0, 2, 1) # [bs, THW, C']
            f = torch.matmul(theta_x, phi_x) # [bs, THW, THW]

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N  # [bs, THW, THW]
        
        y = torch.matmul(f_div_C, g_x) # [bs, THW, C]
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous() # [bs, C, THW]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # [bs, C', T, H, W]
        
        W_y = self.W_z(y)  # [bs, C, T, H, W]
        # residual connection
        z = W_y + x #  # [bs, C, T, H, W]

        # add LayerNorm
        z =  z.permute(0, 2, 3, 4, 1) # [bs, T, H, W, C]
        z = self.norm_layer(z)
        z = z.permute(0, 4, 1, 2, 3) # [bs, C, T, H, W]
        
        return z, audio_temp


