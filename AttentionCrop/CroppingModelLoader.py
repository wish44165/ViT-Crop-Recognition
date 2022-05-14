from cmath import nan
from numpy import arange, dtype
import torch
from AttentionCrop.utils.data_utils import Crop_Divisible_By_N
from AttentionCrop.utils.model import CroppingModel
from torchvision import transforms

class CroppingModelLoader:
    def __init__(self, 
                 dataset, 
                 checkpoint, 
                 device,
                 max_batch_size,
                 shuffle=False,
                 positive_sample_threshold=0.0,
                 patch_len=384,
                 list_downsample_rate=[4, 4, 4, 3, 2],
                 hidden_activation='Mish',
                 return_resized_original_image=False):
        # Initialize some instances 
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.max_batch_size = max_batch_size
        self.positive_sample_threshold = positive_sample_threshold
        self.patch_len = patch_len
        self.preprocess_divisible_by_N = Crop_Divisible_By_N(patch_len)
        self.return_resized_original_image = return_resized_original_image

        if return_resized_original_image:
            self.resize_original = transforms.Resize((patch_len, patch_len))

        # Initialize the cropping model
        cfg = {
            'model': {
                'hidden_activation': hidden_activation,
                'list_downsample_rate': list_downsample_rate
            }
        }
        self.cropping_model = CroppingModel(cfg)
        self.cropping_model = self.cropping_model.to(self.device)

        # Load the model weight
        checkpoint = torch.load(checkpoint)
        self.cropping_model.load_state_dict(checkpoint['model_state_dict'])

    def __iter__(self):
        if self.shuffle:
            sample_idx_list = torch.randperm(len(self.dataset), dtype=torch.int)
        else:
            sample_idx_list = torch.arange(len(self.dataset), dtype=torch.int)

        # Iterate over sample_idx_list and use the index to get a data
        for idx in sample_idx_list:
            img, label = self.dataset.__getitem__(idx)
            img = torch.unsqueeze(self.preprocess_divisible_by_N(img), 0)
            img, label = (img.to(self.device), label.to(self.device))

            with torch.no_grad():
                # Get the prediction
                prediction = self.cropping_model(img)
                while len(prediction.shape) < 3:
                    prediction = torch.unsqueeze(prediction, 0)

                # Add the patch to returning_data by the indices received by get_returning_index()
                indices_h, indices_w, end_index = self.get_returning_index(prediction) 
                start_h_index_list = indices_h * self.patch_len
                end_h_index_list = start_h_index_list + self.patch_len
                start_w_index_list = indices_w * self.patch_len
                end_w_index_list = start_w_index_list + self.patch_len
                for idx, (start_h_index, end_h_index, start_w_index, end_w_index) in enumerate(zip(start_h_index_list, end_h_index_list, start_w_index_list, end_w_index_list)):
                    if idx >= end_index:
                        break
                    if idx == 0:
                        returning_data = img[:, :, start_h_index:end_h_index, start_w_index:end_w_index]
                    else:
                        returning_data = torch.concat((returning_data, img[:, :, start_h_index:end_h_index, start_w_index:end_w_index]), dim=0)
                if self.return_resized_original_image:
                    resized_original_image = self.resize_original(img)
                    yield (returning_data, label, resized_original_image)
                else:
                    yield (returning_data, label)

    def get_returning_index(self, prediction):
        b, h, w = prediction.shape
        topk = torch.topk(prediction.view(-1), min(self.max_batch_size, torch.numel(prediction)))
        topk_values = topk.values
        topk_indices = topk.indices
        ############### Get the index of the positive sample ###############
        # Create a boolean mask ending by `False`, which can avoid picking end_index=0 when all values are `True`.
        thres_mask = topk_values>self.positive_sample_threshold 
        thres_mask = torch.cat((thres_mask, torch.zeros(1, dtype=torch.bool, device=self.device)))
        # When all the samples have an attention score lower than the threshold, keep at least one with the highest score
        end_index = torch.clamp(torch.min(thres_mask, 0)[1], min=1)
        positive_topk_2d_indices_h = torch.div(topk_indices, w, rounding_mode='trunc')
        positive_topk_2d_indices_w = topk_indices % w
        return (positive_topk_2d_indices_h, positive_topk_2d_indices_w, end_index)

if __name__ == "__main__":
    t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    topk = torch.topk(t.view(-1), 4).values
    topk_ = topk[topk > 7]
    print()