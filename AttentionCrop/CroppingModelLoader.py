from numpy import arange, dtype
import torch
from AttentionCrop.utils.data_utils import Crop_Divisible_By_N
from AttentionCrop.utils.model import CroppingModel

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
                 hidden_activation='Mish'):
        # Initialize some instances 
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.max_batch_size = max_batch_size
        self.positive_sample_threshold = positive_sample_threshold
        self.patch_len = patch_len
        self.preprocess_divisible_by_N = Crop_Divisible_By_N(patch_len)

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
                returning_patches = list()

                # Add the patch to returning_patches by the indices received by get_returning_index()
                indices_h, indices_w = self.get_returning_index(prediction)
                for (h, w) in zip(indices_h, indices_w):
                    start_h_index = self.patch_len * h
                    end_h_index = self.patch_len * h + self.patch_len
                    start_w_index = self.patch_len * w
                    end_w_index = self.patch_len * w + self.patch_len
                    returning_patches.append(torch.unsqueeze(img[0, :, start_h_index:end_h_index, start_w_index:end_w_index], 0))
                # Concate the returning patches to a batch of data
                returning_data = torch.concat(returning_patches, dim=0)
                yield (returning_data, label)

    def get_returning_index(self, prediction):
        b, h, w = prediction.shape
        topk = torch.topk(prediction.view(-1), min(self.max_batch_size, torch.numel(prediction)))
        topk_values = topk.values
        topk_indices = topk.indices
        # Get the index of the positive sample
        positive_topk_indices = topk_indices[topk_values > self.positive_sample_threshold]
        # Prevent returning an empty list
        if positive_topk_indices.nelement() == 0:
            positive_topk_indices = torch.Tensor([topk_indices[0]]).to(torch.int)
        positive_topk_2d_indices_h = torch.div(positive_topk_indices, w, rounding_mode='trunc')
        positive_topk_2d_indices_w = positive_topk_indices % w
        return (positive_topk_2d_indices_h, positive_topk_2d_indices_w)

if __name__ == "__main__":
    t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    topk = torch.topk(t.view(-1), 4).values
    topk_ = topk[topk > 7]
    print()