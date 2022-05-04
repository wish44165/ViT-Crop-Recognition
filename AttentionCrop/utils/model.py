import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class Activation(nn.Module):
    def __init__(self, act_func_type):
        super(Activation, self).__init__()
        if act_func_type == 'LeackyReLU':
            self.act_func = nn.LeakyReLU()
        elif act_func_type == 'Mish':
            self.act_func = nn.Mish()
    def forward(self, x):
        return self.act_func(x)

class CroppingModel(nn.Module):
    def __init__(self, cfg):
        super(CroppingModel, self).__init__()
        self.backbone = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=32,                      # model output channels (number of classes in your dataset)
            activation=None
        )
        hidden_activation = cfg['model'].get('hidden_activation', 'LeackyReLU')
        self.activate_backbone = Activation(hidden_activation)

        # Initailize the downsampling model
        list_downsample_rate = cfg['model']['list_downsample_rate']
        sequence = list()
        length_list = len(list_downsample_rate)
        for i, down_rate in enumerate(list_downsample_rate):
            is_lastlayer = True if i == length_list-1 else False
            sequence.append(DownSample(32, down_rate, hidden_activation, is_lastlayer))
        self.down_sample_model = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.backbone(x)
        x = self.activate_backbone(x)
        y = torch.unsqueeze(torch.squeeze(self.down_sample_model(x)), dim=0)
        return y

class DownSample(nn.Module):
    def __init__(self, channel_num, downsample_rate, activation_hidden, is_lastlayer):
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
                Activation(activation_hidden)
            )

    def forward(self, x):
        y = self.sequence(x)
        return y
