# ViT - Crop Recognition
農地作物現況調查影像辨識競賽 - 春季賽:AI作物影響判釋

### [Competitoin Link](https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340)
### [Google Drive](https://drive.google.com/drive/folders/1dOIBsU-zn1JYotF7JEbUyBPG6o2qImyy)
### [HackMD](https://hackmd.io/@x-eSC_X5SMuQbmfwqUOwsQ/ryaRo42Mq)


## Setup


<details>

<summary>Conda environment</summary>
  
```bash
conda create -n Crop python==3.9 -y
conda activate Crop
```

</details>




<details>

<summary>Clone Repository</summary>
  
```bash
git clone https://github.com/TW-yuhsi/ViT-Crop-Recognition.git
pip install -r requirements.txt
```

</details>




<details>

<summary>A PyTorch Extension (Apex)</summary>
  
```bash
git clone https://github.com/NVIDIA/apex
cd apex/
python setup.py install
```

</details>




<details>

<summary>Get pretrained weight</summary>
  
```bash
cd ~ ViT-Crop-Recognition/
mkdir checkpoint
cd checkpoint/
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
  
## Usage
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
### imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

### imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz

```

</details>




<details>

<summary>Get Data</summary>

#### Stage 1 dataset
  
- [Train Data 下載點一](http://aicup-dataset.aidea-web.tw:18080/dataset/train/)
- [Train Data 下載點二 (Google Drive)](https://drive.google.com/drive/folders/1hLNSr6YUIWo8jJO-6BAtZhKP_Ie9wwKw)
- [Train Data 下載點三 (OneDrive)](https://ncku365-my.sharepoint.com/personal/p66101087_ncku_edu_tw/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fp66101087%5Fncku%5Fedu%5Ftw%2FDocuments%2FAICUP%20%E8%A8%93%E7%B7%B4%E8%B3%87%E6%96%99%E9%9B%86&ga=1)
- [Train Data 下載點四 (學術網路)](https://dgl.synology.me/drive/d/s/nzyNfheVlMy6UeKwv0wP8SLAA4T1caMN/hS3Bh_72ez_cWY2PzLaOwMkOkPTSwj0a-_7EgGP3KZAk)
- [Test Data 下載點一](http://aicup-dataset.aidea-web.tw:18080/dataset/test/)
- [Test Data 下載點二 (Google Drive)](https://drive.google.com/drive/folders/1bUP0voNju94nQd_kgr-kr-tjP4opj1Zd?usp=sharing)
- [Test Data 下載點三 (OneDrive)](https://ncku365-my.sharepoint.com/:f:/g/personal/p66101087_ncku_edu_tw/Em_JRy7IcJhIk-Nbrn7Ii34BSh8N4AlX6GyUzg_IboJoKA)
- [Test Data 下載點四 (學術網路)](https://dgl.synology.me/drive/d/s/odkYeGEacoKxd0oTC5BGPjff97OcvHNk/_24MEXkmwEX647b4CcRt8TqsiN6xhv4U-RrZAhjyfgwk)
- [submission_example.csv](http://aicup-dataset.aidea-web.tw:18080/dataset/test/submission_example.csv)
  

</details>




<details>

<summary>Useful commands</summary>
  
```bash=
unzip \*.zip    # Unzip all ZIP files
ls -l | grep "^-" | wc -l    # Check the number of files
```

</details>




## Folder Structure
```
├── data/
│   ├── fold1/
│   │   ├── test/
│   │   │   ├── banana/ bareland/ carrot/ ...
│   │   ├── train/
│   │   │   ├── banana/ bareland/ carrot/ ...
│   │   ├── val/
│   │   │   ├── banana/ bareland/ carrot/ ...
├── ViT-Crop-Recognition/
│   ├── apex/
│   ├── checkpoint/
│   ├── models/
│   ├── requirements.txt
│   ├── train.py
│   ├── utils
```



## Train

Users can split the dataset into `train`, `val`, and `test` folders, as shown in the [Folder Structure](#folder-structure) section in advance, or use the un-split dataset with the CSV files recording the train/val data.

1. The training command using the split dataset:
```bash=
python3 train.py --name Crop --dataset Crop --train_batch_size 8 --img_size 384 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
```
2. The training command using the un-split dataset with the CSV files:
```bash=
python3 train.py --name Crop \
                 --train_batch_size 8 \
                 --img_size 384 \
                 --model_type ViT-B_16 \
                 --pretrained_dir checkpoint/ViT-B_16.npz \
                 --fp16 --fp16_opt_level O2 \
                 --dataset Crop_CSV \
                 --path_csv_train <path of csv_train> \
                 --path_csv_val <path of csv_val>  \
                 --dir_dataset <directory of the dataset> 
```

## Inference

Our ViT model takes images with size 384x384 as inputs, so we have to make the testing images fit this size during the inference phase. In this project, we implement the following two kinds of preprocessing methods to achieve the target:
1. `transforms.Resize()`: adopt the PyTorch resizing function.
```bash
python3 test.py --model_type ViT-B_16 --checkpoint output/Crop_ViT-B_16_checkpoint.bin --img_size 384
```

2. Cropping model: use the cropping model to pick up at most `args.cropping_max_batch_size` critical patches from an input image and concat them into a batch of data in size BxCxHxW.
```bash
python3 test.py --model_type ViT-B_16 \
                --checkpoint ./checkpoint/yuhsi_ViTB16_val_98.73_test_98.64.bin \
                --img_size 384 \
                
                --use_cropping_model \
                --cropping_max_batch_size <the maximum output sample number of the cropping model> \
                --cropping_model_checkpoint <the path to the cropping model checkpoint> \
                --path_csv <the path to the CSV file recording the testing data> \
```

## Ensemble
```bash=
python test_ensemble.py --model_type ["ViT-B_16","ViT-B_16"] --checkpoint ["results/fold1/ViT-B_16_1/Crop_ViT-B_16_checkpoint.bin","results/fold1/ViT-B_16_2/Crop_ViT-B_16_checkpoint.bin"] --img_size [384,384]
```






## Experimental results




<details>

<summary>Toy example</summary>

  
<table>
  <tr>
    <td>Checkpoint</td>
    <td>Model</td>
    <td>Pretrained</td>
    <td>Dataset</td>
    <td>Batch size</td>
    <td>Epochs</td>
    <td>Loss</td>
    <td>Optimizer</td>
    <td>Scheduler</td>
    <td>Augmentation</td>
    <td>Best val epoch</td>
    <td>Best val acc (%)</td>
    <td>test acc (%)</td>
    <td>training time</td>
  </tr>
  <tr>
    <td>2022-04-09</td>
    <td>ResNet101</td>
    <td>imagenet</td>
    <td>1K</td>
    <td>32</td>
    <td>60</td>
    <td>CE</td>
    <td>optimizer = optim.SGD(model.parameters(), lr=0.05)</td>
    <td>step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)</td>
    <td>RandomResizedCrop(416),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))</td>
    <td> </td>
    <td>84 </td>
    <td> </td>
    <td>2hr 41min </td>
  </tr>

</table>


</details>




<details>

<summary>Training Phase</summary>

  
<table>
  <tr>
    <td>Checkpoint</td>
    <td>ID</td>
    <td>Model</td>
    <td>Pretrained</td>
    <td>Dataset</td>
    <td>Batch size</td>
    <td>Epochs</td>
    <td>Loss</td>
    <td>Optimizer</td>
    <td>Scheduler</td>
    <td>Augmentation</td>
    <td>Best val epoch</td>
    <td>Best val acc (%)</td>
    <td>test acc (%)</td>
    <td>training time</td>
  </tr>
  <tr>
    <td>2022-04-18</td>
    <td>1</td>
    <td>ResNet101</td>
    <td>imagenet</td>
    <td>fold1</td>
    <td>16</td>
    <td>1</td>
    <td>CE</td>
    <td>optimizer = optim.SGD(model.parameters(), lr=0.05)</td>
    <td>step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)</td>
    <td>RandomResizedCrop(416),<br>Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))</td>
    <td> </td>
    <td>86.05 </td>
    <td>77.18 </td>
    <td>>1hr </td>
  </tr>
  <tr>
    <td>2022-04-21</td>
    <td>2</td>
    <td>ViT-B_16</td>
    <td>imagenet21k</td>
    <td>fold4</td>
    <td>8</td>
    <td>10000 (iter)</td>
    <td>CE</td>
    <td>optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)</td>
    <td>scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)</td>
    <td>RandomResizedCrop(384)</td>
    <td>9900 (iter)</td>
    <td>97.70 </td>
    <td>97.81 </td>
    <td>9hr 3min </td>
  </tr>
  <tr>
    <td>2022-04-27</td>
    <td>3</td>
    <td>ViT-B_16</td>
    <td>imagenet21k</td>
    <td>fold1</td>
    <td>8</td>
    <td>40000 (iter)</td>
    <td>CE</td>
    <td>optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)</td>
    <td>scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)</td>
    <td>RandomResizedCrop(384)</td>
    <td>31200 (iter)</td>
    <td>98.73 </td>
    <td>98.64 </td>
    <td>1d 13hr 50min </td>
  </tr>
  <tr>
    <td>2022-04-29</td>
    <td>4</td>
    <td>ViT-B_16</td>
    <td>imagenet21k+imagenet2012</td>
    <td>fold1</td>
    <td>8</td>
    <td>40000 (iter)</td>
    <td>CE</td>
    <td>optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)</td>
    <td>scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)</td>
    <td>RandomResizedCrop(384)</td>
    <td>38500 (iter)</td>
    <td>98.74 </td>
    <td>98.74 </td>
    <td>1d 13hr 57min </td>
  </tr>
  <tr>
    <td>2022-05-03</td>
    <td>5</td>
    <td>ViT-B_16</td>
    <td>imagenet21k+imagenet2012</td>
    <td>fold1</td>
    <td>8</td>
    <td>10000 (iter)</td>
    <td>CE</td>
    <td>optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)</td>
    <td>scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)</td>
    <td>RandomResizedCrop(384)</td>
    <td>9400 (iter)</td>
    <td>98.16 </td>
    <td>98.07 </td>
    <td>9hr 23min </td>
  </tr>

</table>


</details>





<details>

<summary>Training Phase Final</summary>
  
Coming ...

</details>




<details>

<summary>Testing Phase</summary>
  
Following results show:
1. The cropping model helps when the maximum batch size is large and with a positive sample threshold close to but greater than zero.
2. Filter some predictions with the entropy higher than the threshold before the internal ensemble helps.
3. Using the testing augmentation only with horizontal flip might help, but we still have to do more experiments.

### Testing Set
  
<table>
  <tr>
    <td>Fold id</td>
    <td>Use cropping model?</td>
    <td>Checkpoint description</td>
    <td>Maximum batch size</td>
    <td>Positive sample threshold</td>
    <td>Entropy filter threshold</td>
    <td>Testing augmentation: rotation</td>
    <td>Testing augmentation: sharpness</td>
    <td>Testing augmentation: flip</td>
    <td>Testing augmentation: jitter</td>
    <td>Accuracy (%)</td>
  </tr>
  <tr>
    <td>1</td>
    <td>False</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.6443</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>8</td>
    <td>0.3</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.5945</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>8</td>
    <td>0.5</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.5945</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>8</td>
    <td>0.7</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.5697</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.8060</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>-0.1</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.7562</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.95</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.8682</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.8930</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.05</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.8930</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>O</td>
    <td>O</td>
    <td>O</td>
    <td>O</td>
    <td>98.7065</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>Nan</td>
    <td>O</td>
    <td>O</td>
    <td>O</td>
    <td>O</td>
    <td>97.2388</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>O</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.8433</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>Nan</td>
    <td>O</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.8433</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>O</td>
    <td>Nan</td>
    <td>98.9801</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>O</td>
    <td>98.7065</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>Nan</td>
    <td>O</td>
    <td>O</td>
    <td>O</td>
    <td>98.6567</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>O</td>
    <td>Nan</td>
    <td>O</td>
    <td>O</td>
    <td>98.7438</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>O</td>
    <td>O</td>
    <td>Nan</td>
    <td>O</td>
    <td>98.6816</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.3</td>
    <td>O</td>
    <td>O</td>
    <td>O</td>
    <td>Nan</td>
    <td>98.8433</td>
  </tr>

</table>
  
### Validation Set
  
<table>
  <tr>
    <td>Fold id</td>
    <td>Use cropping model?</td>
    <td>Checkpoint description</td>
    <td>Maximum batch size</td>
    <td>Positive sample threshold</td>
    <td>Entropy filter threshold</td>
    <td>Accuracy (%)</td>
  </tr>
  <tr>
    <td>1</td>
    <td>False</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>Nan</td>
    <td>98.7354</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>8</td>
    <td>0.3</td>
    <td>Nan</td>
    <td>98.6110</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>8</td>
    <td>0.5</td>
    <td>Nan</td>
    <td>98.6110</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>8</td>
    <td>0.7</td>
    <td>Nan</td>
    <td>98.5972</td>
  </tr>
  <tr>
    <td>1</td>
    <td>True</td>
    <td>Unet-Mish-ch64-4^3*3*2/iteration_100000.pth</td>
    <td>36</td>
    <td>0.0</td>
    <td>0.05</td>
    <td>98.8870</td>
  </tr>

</table>

</details>




## GitHub Acknowledgement
- Augmentation
  - AutoAugment: https://github.com/DeepVoltaire/AutoAugment
  - TTAch: https://github.com/qubvel/ttach
- Optimizer
  - Ranger: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
  - Ranger21: https://github.com/lessw2020/Ranger21 
  - SAM: https://github.com/davda54/sam
- Loss function
  - MCCE: https://github.com/Kurumi233/Mutual-Channel-Loss
  - FLSD: https://github.com/torrvision/focal_calibration




## Reference
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)




## Citation
```
@article{
    title  = {Crop classification},
    author = {Yu-Hsi Chen, Jiun-Han Chen, Yu-Kai Chen, Kuan-Wei Zeng, Yang Tseng},
    url    = {https://github.com/TW-yuhsi/ViT-Crop-Recognition},
    year   = {2022}
}
```
