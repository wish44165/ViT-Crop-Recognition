# ViT - Crop Recognition
農地作物現況調查影像辨識競賽 - 春季賽:AI作物影響判釋



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

</details>




<details>

<summary>Get Data</summary>
  
- [Train Data 下載點一](http://aicup-dataset.aidea-web.tw:18080/dataset/train/)
- [Train Data 下載點二 (Google Drive)](https://drive.google.com/drive/folders/1hLNSr6YUIWo8jJO-6BAtZhKP_Ie9wwKw)
- [Train Data 下載點三 (OneDrive)](https://ncku365-my.sharepoint.com/personal/p66101087_ncku_edu_tw/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fp66101087%5Fncku%5Fedu%5Ftw%2FDocuments%2FAICUP%20%E8%A8%93%E7%B7%B4%E8%B3%87%E6%96%99%E9%9B%86&ga=1)
- [Train Data 下載點四 (學術網路)](https://dgl.synology.me/drive/d/s/nzyNfheVlMy6UeKwv0wP8SLAA4T1caMN/hS3Bh_72ez_cWY2PzLaOwMkOkPTSwj0a-_7EgGP3KZAk)
- [Test Data (5/14開放)]()

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
```bash=
python3 train.py --name Crop --dataset Crop --train_batch_size 8 --img_size 384 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
```




## Inference
```bash=
python3 test.py --model_type ViT-B_16 --checkpoint output/Crop_ViT-B_16_checkpoint.bin --img_size 384
```



## [Google Drive](https://drive.google.com/drive/folders/1dOIBsU-zn1JYotF7JEbUyBPG6o2qImyy)
## [HackMD](https://hackmd.io/@x-eSC_X5SMuQbmfwqUOwsQ/ryaRo42Mq)




## Experimental results




<details>

<summary>Toy example</summary>

  
<table>
  <tr>
    <td>Checkpoint</td>
    <td>Model</td>
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
    <td>2022-04-07</td>
    <td>ResNet101</td>
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
    <td>Model</td>
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
    <td>2022-04-14</td>
    <td>ResNet101</td>
    <td>fold4</td>
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
    <td>2022-04-17</td>
    <td>ViT-B_16</td>
    <td>fold4</td>
    <td>8</td>
    <td>10000 (iter)</td>
    <td>CE</td>
    <td>optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)</td>
    <td>scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)</td>
    <td>RandomResizedCrop(416)</td>
    <td>9900 (iter)</td>
    <td>97.70 </td>
    <td>97.81 </td>
    <td>>9hr 3min </td>
  </tr>

</table>


</details>





<details>

<summary>Training Phase Final</summary>
  
Coming ...

</details>




<details>

<summary>Testing Phase</summary>
  
Coming ...

</details>




## GitHub Acknowledgement
- Augmentation
  - AutoAugment: https://github.com/DeepVoltaire/AutoAugment
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
    author = {Yu-Hsi Chen, Kuan-Wei Zeng, Yu-Kai Chen, Tseng Yang},
    url    = {https://github.com/TW-yuhsi/ViT-Crop-Recognition},
    year   = {2022}
}
```
