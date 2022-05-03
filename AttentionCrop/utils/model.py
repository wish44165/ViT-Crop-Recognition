import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CroppingModel(nn.Module):
    def __init__(self):
        super(CroppingModel, self).__init__()
        self.backbone = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=32,                      # model output channels (number of classes in your dataset)
            activation=None
        )
        self.activate_backbone = nn.LeakyReLU()

        # Initailize the downsampling model
        list_downsample_rate = [2, 2, 2, 2, 2, 2, 2, 3]
        sequence = list()
        length_list = len(list_downsample_rate)
        for i, down_rate in enumerate(list_downsample_rate):
            is_lastlayer = True if i == length_list-1 else False
            sequence.append(DownSample(32, down_rate, is_lastlayer))
        self.down_sample_model = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.backbone(x)
        x = self.activate_backbone(x)
        y = torch.unsqueeze(torch.squeeze(self.down_sample_model(x)), dim=0)
        return y

class DownSample(nn.Module):
    def __init__(self, channel_num, downsample_rate, is_lastlayer):
        super(DownSample, self).__init__()
        if is_lastlayer:
            self.sequence = nn.Sequential(
                nn.Conv2d(channel_num, 1, downsample_rate, downsample_rate),
                nn.Tanh()
            )
        else:
            self.sequence = nn.Sequential(
                nn.Conv2d(channel_num, channel_num, downsample_rate, downsample_rate),
                nn.InstanceNorm2d(channel_num),
                nn.LeakyReLU()
            )

    def forward(self, x):
        y = self.sequence(x)
        return y
