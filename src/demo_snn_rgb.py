# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath('./'))
from src.snn_model import Spiking_NeuLF_snn, Spiking_NeuLF_ann
from NeuLF.src.utils import rm_folder, rm_folder_keep
import cv2,glob
import torchvision
from NeuLF.src.cam_view import rayPlaneInter
from tqdm import tqdm
from NeuLF.src.utils import eval_uvst
from NeuLF.src.utils import get_rays_np
import imageio
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser() # museum,column2
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--data_dir',type=str, 
                    default = 'dataset/Ollie/',help='data folder name')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--scale', type=int, default = 4)
parser.add_argument('--img_form',type=str, default = '.png',help = 'exp name')
parser.add_argument('--time_steps',type=int, default = 500,help = 'time steps')

class demo_SN_rgb():
    def __init__(self,args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        scale = 22
        # data_root
        data_root = args.data_dir
        self.snn_model = Spiking_NeuLF_snn(D=args.mlp_depth, time_steps=args.time_steps, scale=scale)
        self.ann_model = Spiking_NeuLF_ann()
        data_img = os.path.join(args.data_dir,'images_{}'.format(args.scale)) 

        self.exp = 'Exp_'+args.exp_name
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'

        self.img_folder_test = 'demo_result_rgb/'+self.exp+'_'+f'{args.time_steps}'+'/'
        rm_folder(self.img_folder_test)

        snn_ckpt_name = f"./{self.checkpoints}/snn_model.pth"
        ann_ckpt_name = f"./{self.checkpoints}/ann_model.pth"

        print(f"Load weights from {snn_ckpt_name}")
        snn_ckpt = torch.load(snn_ckpt_name)
        print(f"Load weights from {ann_ckpt_name}")
        ann_ckpt = torch.load(ann_ckpt_name)
        self.snn_model = self.snn_model.cuda()
        self.snn_model.load_state_dict(snn_ckpt, strict=False)
        with torch.no_grad():
            for name, module in self.snn_model.named_modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data /= scale

        self.ann_model = self.ann_model.cuda()
        self.ann_model.load_state_dict(ann_ckpt, strict=False)

        # height and width
        image_paths = glob.glob(f"{data_img}/*"+args.img_form)
        sample_img = cv2.imread(image_paths[0])
        self.h = int(sample_img.shape[0])
        self.w = int(sample_img.shape[1])

        # get max 
        self.uvst_whole   = np.load(f"{data_root}/uvsttrain.npy") 
        self.uvst_min = self.uvst_whole.min()
        self.uvst_max = self.uvst_whole.max()
        self.color_whole   = np.load(f"{data_root}/rgbtrain.npy")

        self.color_imgs = np.reshape(self.color_whole,(-1,self.h,self.w,3))
        # input val
        

        self.fdepth = np.load(f"{data_root}/fdepthtrain.npy") # center object

        rays_whole = np.concatenate([self.uvst_whole, self.color_whole], axis=1)
        self.min_u,self.max_u,self.min_v,self.max_v,self.min_s,self.max_s,self.min_t,self.max_t = eval_uvst(rays_whole)
       
        self.uv_depth     = 0.0
        self.st_depth     = -self.fdepth
       
        # render pose
        self.render_pose  = np.load(f"{data_root}/Render_posetrain.npy")#render path spiral
        self.intrinsic    = np.load(f"{data_root}/ktrain.npy")

        # load camera pose
        self.camera_pose = np.load(f"{data_root}/cam_posetrain.npy") 
        print(self.camera_pose.shape[0])

        # find nearest view in training
        self.knn = NearestNeighbors(n_neighbors=1)
        self.knn.fit(self.camera_pose)

    def gen_pose_llff(self):
        
        savename = f"{self.img_folder_test}/llff_pose"

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 30.0, (self.w,self.h))
        out_near   = cv2.VideoWriter(savename+"_near.mp4", fourcc, 30.0, (self.w,self.h))

        view_group       = []
        view_group_depth = []
        view_group_near  = []

        num_frame = self.render_pose.shape[0]

        with torch.no_grad():
            cnt = 0
            for i, c2w in enumerate(tqdm(self.render_pose)):
                ray_o, ray_d = get_rays_np(self.h, self.w, self.intrinsic, c2w)

                # inference view camera translation
                current_T = c2w[:3,-1].T
                current_T = np.expand_dims(current_T,axis=0)

                distance,neighbor_cam_index = self.knn.kneighbors(current_T)
                neighbor_cam_index = neighbor_cam_index.item()

                ray_o = np.reshape(ray_o,(-1,3))
                ray_d = np.reshape(ray_d,(-1,3))

                plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

                p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
                p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))

                # interset radius plane 
                inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)
                inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)

                data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)

                data_uvst = (data_uvst - self.uvst_min)/(self.uvst_max - self.uvst_min)
                save_path = savename + '.png'
                cnt += 1
                view_unit = self.render_sample_img(cnt=cnt,snn_model=self.snn_model,ann_model=self.ann_model,uvst=data_uvst,w=self.w,h=self.h,save_path=save_path,save_flag=True)
                
                # view_unit_near = 0
                view_unit_near = self.color_imgs[neighbor_cam_index,:,:,:].copy()

                # unit
                view_unit *= 255
                view_unit_near *=255
           
                view_unit       = view_unit.cpu().numpy().astype(np.uint8)
                view_unit_near = view_unit_near.astype(np.uint8)

                out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
                out_near.write(cv2.cvtColor(view_unit_near,cv2.COLOR_RGB2BGR))
                
                view_unit       = imageio.core.util.Array(view_unit)
                view_unit_near = imageio.core.util.Array(view_unit_near)
                
                view_group.append(view_unit)
                view_group_near.append(view_unit_near)

            imageio.mimsave(savename+".gif", view_group,fps=30)
            imageio.mimsave(savename+"_near.gif", view_group_near,fps=30)

    def render_sample_img(self,cnt,snn_model,ann_model,uvst, w, h, save_path=None,save_flag=True):
         with torch.no_grad():
        
            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()

            snn_output = snn_model(uvst)
            pred_color = ann_model(snn_output)
  
            pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))

            if(save_flag) & (cnt == 1):
                torchvision.utils.save_image(pred_img, save_path)
            
            return pred_color.reshape((h,w,3)) #,pred_depth_norm.reshape((h,w,1))
        

if __name__ == '__main__':

    args = parser.parse_args()

    unit = demo_SN_rgb(args)
    unit.gen_pose_llff()
 
