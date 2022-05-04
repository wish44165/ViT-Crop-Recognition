import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_utils import get_cropping_model_loader
from utils.model import CroppingModel

import os
import cv2
import yaml
import argparse
import numpy as np

class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = CroppingModel(cfg)
        self.return_file_name = True if cfg['test']['do_visualization'] else False
        self.dataloader = get_cropping_model_loader(cfg, self.return_file_name, is_test=True)
        self.loss_function = nn.MSELoss()
        self.device = 'cuda'
        self.downsample_rate = cfg['test']['visualization_downsample_rate']

        class_name_list = ['banana', 'bareland', 'carrot', 'corn', 'dragonfruit', 'garlic', 'guava', 'inundated', 'peanut', 'pineapple', 'pumpkin', 'rice', 'soybean', 'sugarcane', 'tomato']
        if cfg['test']['do_visualization']:
            root_folder_name = os.path.join(cfg['test']['output']['folder'], cfg['test']['output']['description'])
            for class_name in class_name_list:
                if not os.path.isdir(os.path.join(root_folder_name, 'correctness_map', class_name)):
                    os.makedirs(os.path.join(root_folder_name, 'correctness_map', class_name))
                if not os.path.isdir(os.path.join(root_folder_name, 'difference_map', class_name)):
                    os.makedirs(os.path.join(root_folder_name, 'difference_map', class_name))

    def start(self):
        self.load_model()
        self.model = self.model.to(self.device)
        total_iters_val = len(self.dataloader.dataset) // self.cfg['val']['batch_size']
        progress_bar = tqdm(enumerate(self.dataloader), total=total_iters_val, desc=f'Validating...')
        total_loss = 0.
        total_correct_num = 0.
        total_sample_num = 0
        with torch.no_grad():
            for i, data in progress_bar:
                # Get the training data and move them to the target device
                if self.return_file_name:
                    (img, label, class_name, file_name) = data
                    class_name = class_name[0]
                    file_name = file_name[0]
                    img, label = (img.to(self.device), label.to(self.device))
                else:
                    img, label = (d.to(self.device) for d in data)
            
                ############# Forward propagation ###############
                prediction = self.model(img)

                ############# Accumulate accuracy ###############
                # Make the shape of prediction to be BxHxW
                if len(prediction.shape) == 2:
                    prediction = torch.unsqueeze(prediction, dim=0)
                # For every prediction in the batch
                for i in range(label.shape[0]):
                    num_correct = torch.eq((prediction[i] > 0), (label[i] > 0)).sum()
                    num_sample = torch.flatten(prediction).size()[0]
                    total_correct_num += num_correct
                    total_sample_num += num_sample
                if self.cfg['test']['do_visualization']:
                    file_name = os.path.join(class_name, file_name)
                    self.visualize(file_name, img, prediction, label)

                ############# Accumulate loss ###############
                loss = self.loss_function(prediction, label)
                total_loss += loss

            ############# Get loss and accuracy ###############
            total_loss /= total_iters_val
            accuracy = total_correct_num / total_sample_num
        print('=================================================')
        print(f'Validation result:')
        print(f'Loss: {total_loss}')
        print(f'Accuracy: {accuracy}\n')

    def load_model(self):
        checkpoint = torch.load(self.cfg['test']['checkpoint'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def visualize(self, filename, img, prediction, label):
        img = torch.squeeze(img).permute(1, 2, 0).cpu().numpy()
        img_ratio_difference = self.cfg['test']['original_img_ratio_difference_map']
        img_ratio_correctness = self.cfg['test']['original_img_ratio_correctness_map']
        upsample_rate = int(self.cfg['data']['classification_model_resolution'] / self.downsample_rate)
        if len(prediction.shape) == 1:
            prediction = torch.unsqueeze(prediction, 0)
        prediction_h, prediction_w = prediction.shape[-2:]
        vis_h, vis_w = prediction_h*upsample_rate, prediction_w*upsample_rate
        img = (np.clip((img*0.5)+0.5, a_min=0, a_max=1)*255).astype(np.uint8)
        img = cv2.resize(img, (vis_w, vis_h), cv2.INTER_NEAREST)
        # Convert the image from RGB to BGR
        img = img[:, :, ::-1]

        ############## Get the visualization of the difference-map ##############
        difference = torch.absolute(label - prediction)
        # Set the difference to the range [0, 255]
        difference = torch.squeeze(difference * 127.5).to(torch.uint8).cpu().numpy()
        if difference.size == 1:
            difference = np.expand_dims(difference, axis=0)
            difference = np.expand_dims(difference, axis=0)
        difference_resized = np.zeros((vis_h, vis_w), dtype=np.uint8)
        for h in range(prediction_h):
            for w in range(prediction_w):
                difference_resized[h*upsample_rate:h*upsample_rate+upsample_rate-1, w*upsample_rate:w*upsample_rate+upsample_rate-1] = difference[h][w]
        difference_resized = cv2.applyColorMap(difference_resized, cv2.COLORMAP_JET)
        if len(prediction.shape) == 2:
            prediction.unsqueeze_(0)
        for h in range(prediction_h):
            for w in range(prediction_w):
                if prediction[0][h][w] < 0:
                    difference_resized[h*upsample_rate:h*upsample_rate+20-1, w*upsample_rate:w*upsample_rate+20-1] = (0, 0, 0)
        difference_map = img_ratio_difference * img + (1 - img_ratio_difference) * difference_resized
        path_difference_map = os.path.join(self.cfg['test']['output']['folder'], 
                                           self.cfg['test']['output']['description'],
                                           'difference_map',
                                           f'{filename}.jpg')
        cv2.imwrite(path_difference_map, difference_map)

        ############## Get the visualization of the correctness-map ##############
        correctness = torch.eq((prediction[0] > 0), (label[0] > 0))
        correctness = torch.squeeze(correctness).cpu().numpy()
        if correctness.size == 1:
            correctness = np.expand_dims(correctness, axis=0)
            correctness = np.expand_dims(correctness, axis=0)
        correctness_resized = np.zeros((vis_h, vis_w), dtype=np.uint8)
        for h in range(prediction_h):
            for w in range(prediction_w):
                correctness_resized[h*upsample_rate:h*upsample_rate+upsample_rate-1, w*upsample_rate:w*upsample_rate+upsample_rate-1] = correctness[h][w]
        correctness_resize_colorized = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        correctness_resize_colorized[:, :] = (0, 69, 255) #(114, 128, 250)
        correctness_resize_colorized[correctness_resized == 1] = (50, 205, 50) #(152,251,152)
        for h in range(prediction_h):
            for w in range(prediction_w):
                if prediction[0][h][w] < 0:
                    correctness_resize_colorized[h*upsample_rate:h*upsample_rate+20-1, w*upsample_rate:w*upsample_rate+20-1] = (0, 0, 0)
        correctness_map = img_ratio_correctness * img + (1 - img_ratio_correctness) * correctness_resize_colorized
        path_correctness_map = os.path.join(self.cfg['test']['output']['folder'], 
                                           self.cfg['test']['output']['description'],
                                           'correctness_map',
                                           f'{filename}.jpg')
        cv2.imwrite(path_correctness_map, correctness_map)

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./config_files/config_train.yaml',
                        type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    trainer = Tester(cfg)
    trainer.start()