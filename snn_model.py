import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import snntorch as snn




class Spiking_NeuLF_snn(nn.Module):
    def __init__(self, beta=0.9, time_steps = 4, D=8, W=256, input_ch=256, output_ch=4, skips=[4,8,12,16,20]):
        """ 
        """
        super(Spiking_NeuLF_snn, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch

        self.skips = np.arange(4, D, 4)
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.L=8
        self.views_linears = nn.ModuleList([nn.Linear(input_ch, W//2)])

        # if use_viewdirs:
        self.feature_linear = nn.Linear(W, W)
        
        self.rgb_linear = nn.Linear(W//2, 3)

        #self.rgb_act   = nn.Sigmoid()                      ## Don't use sigmoid function for SNN

        self.input_net =  nn.Linear(4, input_ch)
       
        self.b = Parameter(torch.normal(mean=0.0, std=3, size=(int(input_ch/2), 4)), requires_grad=False)

        self.input_net_pe =  nn.Linear(self.L*8, input_ch)

        self.time_steps = time_steps

        self.lif_in = snn.Leaky(beta=beta)

        self.lif0 = snn.Leaky(beta=beta)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif3 = snn.Leaky(beta=beta)
        self.lif4 = snn.Leaky(beta=beta)
        self.lif5 = snn.Leaky(beta=beta)
        self.lif6 = snn.Leaky(beta=beta)
        self.lif7 = snn.Leaky(beta=beta)

        self.lif_view = snn.Leaky(beta=beta)

        self.spk_in_rec = []

        self.spk0_rec = []
        self.spk1_rec = []
        self.spk2_rec = []
        self.spk3_rec = []
        self.spk4_rec = []
        self.spk5_rec = []
        self.spk6_rec = []
        self.spk7_rec = []
        
        self.spk_view_rec = []

    def forward(self, x):
        for t in range(self.time_steps):
            input_pts = self.input_net(x)

            #input_pts = F.relu(input_pts)      # Original Code
            input_pts = self.lif_in(input_pts)

            h = input_pts
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                if i == 0:
                    h = self.lif0(h)
                elif i == 1:
                    h = self.lif1(h)
                elif i == 2:
                    h = self.lif2(h)
                elif i == 3:
                    h = self.lif3(h)
                elif i == 4:
                    h = self.lif4(h)
                elif i == 5:
                    h = self.lif5(h)
                elif i == 6:
                    h = self.lif6(h)
                elif i == 7:
                    h = self.lif7(h)

                if i in self.skips:
                    h = torch.cat([h, input_pts], -1)

            feature = self.feature_linear(h)
            h = feature

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = self.lif_view(h)

            #rgb = self.rgb_linear(h)                   ## For SNN, Model must be finsished at ReLU.

            #rgb   = self.rgb_act(rgb)                  ## Don't use sigmoid function for SNN
        

        return h

class Spiking_NeuLF_ann(nn.Module):
    def __init__(self, W):
        super(Spiking_NeuLF_ann, self).__init__()

        self.W = W

        self.rgb_linear = nn.Linear(W//2, 3)
        self.rgb_act   = nn.Sigmoid()
    
    def forward(self, h):

        rgb = self.rgb_linear(h)
        rgb = self.rgb_act(rgb)