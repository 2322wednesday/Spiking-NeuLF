import torch
from model import Nerf4D_relu_ps
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256_0to1',help = 'exp name')

class extract_weights():
    def __init__(self, args):
        
        self.ckpt_path = f"result/Exp_{args.exp_name}/checkpoints/nelf-100.pth"
        self.save_dir = f"result/Exp_{args.exp_name}/checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

        self.ckpt = torch.load(self.ckpt_path, map_location="cpu")
    
    
    def extract(self,args):
        base_model = Nerf4D_relu_ps(D=8, W=256, input_ch=256)
        base_model.load_state_dict(self.ckpt)
        full_state = base_model.state_dict()

        snn_keys = [k for k in full_state.keys() if not k.startswith("rgb_linear") and not k.startswith("depth_linear")]
        snn_state = {k: full_state[k] for k in snn_keys}
        torch.save(snn_state, f"{self.save_dir}/snn_model.pth")

        ann_keys = [k for k in full_state.keys() if k.startswith("rgb_linear")]
        ann_state = {k: full_state[k] for k in ann_keys}
        torch.save(ann_state, f"{self.save_dir}/ann_model.pth")

        print("SNN and ANN weights extracted successfully.")

if __name__ == '__main__':
    args = parser.parse_args()
    ext_w = extract_weights(args)
    ext_w.extract(args)
