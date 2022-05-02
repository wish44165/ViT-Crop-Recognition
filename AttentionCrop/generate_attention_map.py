import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms.functional as F

import sys
sys.path.append('../')
from utils.data_utils import get_attn_loader
from models.modeling import VisionTransformer, CONFIGS

class GenerateAttnMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataloader = get_attn_loader(cfg)
        config = CONFIGS[cfg['train']['model']['type']]
        self.model = VisionTransformer(config, cfg['train']['img-size'], zero_head=True, num_classes=15, vis=True)
        self.model.load_state_dict(torch.load(cfg['directory']['checkpoint']))
        self.device = cfg['train']['device'] 
        self.model = self.model.to(self.device)
        self.dir_vis = os.path.join(cfg['directory']['output']['root-dir'], cfg['directory']['output']['sub-dir-vis'])
        self.dir_attn_map = os.path.join(cfg['directory']['output']['root-dir'], cfg['directory']['output']['sub-dir-attention-map'])
        for class_name in self.dataloader.dataset.className2idx.keys():
            if not os.path.isdir(os.path.join(self.dir_vis, class_name)):
                os.makedirs(os.path.join(self.dir_vis, class_name))
            if not os.path.isdir(os.path.join(self.dir_attn_map, class_name)):
                os.makedirs(os.path.join(self.dir_attn_map, class_name))

    def process_small_patch(self, x, label):
        im = (torch.squeeze(torch.clamp(x*0.5+0.5, min=-1., max=1.))*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        logits, att_mat = self.model(x)

        att_mat = torch.stack(att_mat).squeeze(1)

        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1)).to(self.device)
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size()).to(self.device)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
        mask = cv2.resize(mask / mask.max(), im.shape[:-1])[..., np.newaxis]

        probs = torch.nn.Softmax(dim=-1)(logits)
        prediction = torch.argmax(probs, dim=-1)
        ratio_mask = 0.5
        mask_vis = (mask*255).astype(np.uint8)
        if prediction == label:
            mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        else:
            mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_BONE)
            mask *= -1
        vis = mask_vis*ratio_mask + im*(1-ratio_mask)
        return vis, mask.sum()

    def process(self):
        with torch.no_grad():
            progress_bar = tqdm(self.dataloader, total=len(self.dataloader.dataset), desc='Processing...')
            for data in progress_bar:
                (img, label, class_name, file_name) = data
                class_name = class_name[0]
                file_name = file_name[0]
                img = img.to(self.device)
                label = label.to(self.device)
                height, width = img.shape[-2:]
                img_size = self.cfg['train']['img-size']
                vis_whole_img = np.zeros((height, width, 3))
                attn_whole_img = np.zeros((height//img_size, width//img_size))
                width_list = [w for w in range(img_size-1, width, img_size)]
                height_list = [h for h in range(img_size-1, height, img_size)]

                for idx_h, h in enumerate(height_list):
                    for idx_w, w in enumerate(width_list):
                        top = h-img_size+1
                        left = w-img_size+1
                        x = F.crop(img, top=top, left=left, height=img_size, width=img_size)
                        vis_patch, attn_patch = self.process_small_patch(x, label)
                        vis_whole_img[top:h+1, left:w+1, :] = vis_patch
                        attn_whole_img[idx_h:idx_h+1, idx_w:idx_w+1] = attn_patch
                vis_whole_img = vis_whole_img[:, :, ::-1]
                cv2.imwrite(os.path.join(self.dir_vis, class_name, file_name+'.jpg'), vis_whole_img)
                attn_whole_img.tofile(os.path.join(self.dir_attn_map, class_name, file_name+'.dat'))

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='config.yaml',
                        type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    GenAttnMap = GenerateAttnMap(cfg)
    GenAttnMap.process()
    