import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import snntorch as snn




class Spiking_NeuLF_snn(nn.Module):
    def __init__(self, beta=0.9, time_steps = 1000, D=8, W=256, input_ch=256, rec=False):
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
        
        self.rec = rec
        if self.rec:
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
        mem_in      = self.lif_in.init_leaky()
        mem0        = self.lif0.init_leaky()
        mem1        = self.lif1.init_leaky()
        mem2        = self.lif2.init_leaky()
        mem3        = self.lif3.init_leaky()
        mem4        = self.lif4.init_leaky()
        mem5        = self.lif5.init_leaky()
        mem6        = self.lif6.init_leaky()
        mem7        = self.lif7.init_leaky()
        mem_view    = self.lif_view.init_leaky()
 
        for t in range(self.time_steps):
            input_pts = self.input_net(x[t])

            #input_pts = F.relu(input_pts)      # Original Code
            input_pts, mem_in = self.lif_in(input_pts, mem_in)
            self.spk_in_rec.append(input_pts)

            h = input_pts
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                if i == 0:
                    h, mem0 = self.lif0(h, mem0)
                    if self.rec:
                        self.spk0_rec.append(h)
                elif i == 1:
                    h, mem1 = self.lif1(h, mem1)
                    if self.rec:    
                        self.spk1_rec.append(h)
                elif i == 2:
                    h, mem2 = self.lif2(h, mem2)
                    if self.rec:    
                        self.spk2_rec.append(h)
                elif i == 3:
                    h, mem3 = self.lif3(h, mem3)
                    if self.rec:    
                        self.spk3_rec.append(h)
                elif i == 4:
                    h, mem4 = self.lif4(h, mem4)
                    if self.rec:    
                        self.spk4_rec.append(h)
                elif i == 5:
                    h, mem5 = self.lif5(h, mem5)
                    if self.rec:    
                        self.spk5_rec.append(h)
                elif i == 6:
                    h, mem6 = self.lif6(h, mem6)
                    if self.rec:    
                        self.spk6_rec.append(h)
                elif i == 7:
                    h, mem7 = self.lif7(h, mem7)
                    if self.rec:    
                        self.spk7_rec.append(h)

                if i in self.skips:
                    h = torch.cat([h, input_pts], -1)

            feature = self.feature_linear(h)
            h = feature

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h, mem_view = self.lif_view(h, mem_view)
                self.spk_view_rec.append(h)

            #rgb = self.rgb_linear(h)                   ## For SNN, Model must be finsished at ReLU.

            #rgb   = self.rgb_act(rgb)                  ## Don't use sigmoid function for SNN
        
        if self.rec:
            spk_in_rec      = torch.stack(self.spk_in_rec)
            spk0_rec        = torch.stack(self.spk0_rec)
            spk1_rec        = torch.stack(self.spk1_rec)
            spk2_rec        = torch.stack(self.spk2_rec)
            spk3_rec        = torch.stack(self.spk3_rec)
            spk4_rec        = torch.stack(self.spk4_rec)
            spk5_rec        = torch.stack(self.spk5_rec)
            spk6_rec        = torch.stack(self.spk6_rec)
            spk7_rec        = torch.stack(self.spk7_rec)
        spk_view_rec    = torch.stack(self.spk_view_rec)

        return spk_view_rec.sum() / self.time_steps

class Spiking_NeuLF_ann(nn.Module):
    def __init__(self, W=256):
        super(Spiking_NeuLF_ann, self).__init__()

        self.W = W

        self.rgb_linear = nn.Linear(W//2, 3)
        self.rgb_act   = nn.Sigmoid()
    
    def forward(self, h):

        rgb = self.rgb_linear(h)
        rgb = self.rgb_act(rgb)