# Cropping Model

This folder is related to the cropping model, including the following features:
1. The program generates ground truth to train the cropping model.
2. The training program: Train the cropping model.
3. CroppingModelLoader: A dataloader utilizes the cropping model to crop several patches from the input images.

## 1. Generate the Ground Truth

Use a `.yaml` file to configure the program.

```yaml
directory:
  checkpoint: ../checkpoint/yuhsi_ViTB16_val_98.73_test_98.64.bin   # The path to the pretrained weight
  data:
    root-dir: ../../data/crop_data/data/                            # The path to the Crop dataset
    path-csv: ../../data/crop_data/seperate_csv/fold1_train.csv     # The path to the CSV file storing the data to be used
  output:
    root-dir: ../output/attention                                   # The path to save the results
    sub-dir-vis: visualization                                      # A sub-folder under <root-dir> storing the visualizations
    sub-dir-attention-map: attention-map                            # A sub-folder under <root-dir> storing the ground truth for training the cropping model

train:
  attention-method: grad-CAM                                        # Currently only implemented GRAD-CAM, this option doesn't have the function
  model:                                                            
    type: ViT-B_16                                                  # Choose from ["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"]
  img-size: 384                                                     # The input resolution of the assigned model
  device: cuda
```

### Execution Command

```bash
python3 generate_attention_map.py --cfg <Config File>
```

## 2. Training And Testing Program

Use a `.yaml` shared by `train.py` and `test.py` file to configure the program.

### Configuration File 

```yaml
output:
  output_folder: 'results/fold1_real'             # The name of the output root folder
  description: Unet-Mish-ch64-4^3*3*2             # The name of the output sub-folder

keep_train: false                                 # Whether keep training from the pre-trained weight
keep_train_obj: ['model', 'optimizer']            # ['model', 'optimizer', 'lr_scheduler']
path_pretrained_weight: /work/kevin8ntust/ViT-Crop-Recognition/AttentionCrop/results/Unet-Mish-ch64-4^3*3*2/iteration_100000.pth

train:
  batch_size: 1
  total_iterations: 200000                        # Total training iterations
  eval_freq: 10000                                # How often to do the evaluation
  save_model_freq: 1000                           # How often to save the model
  lr_scheduler: 
    sched: cosine
    warmup_iterations: 3332         
    warmup_lr: 0.000001
    min_lr: 0.00001
  optimizer: 
    mode: adamW
    lr: 0.00025
    weight_decay: 0.1
val:
  batch_size: 1
data:
  classification_model_resolution: 384                                    # The input resolution of the classification model
  root_dataset_img: ../../data/crop_data/data/                            # The root directory of the crop images
  root_dataset_attn: ../output/attention/attention-map/                   # The root directory of the generated attention map ground truth
  path_csv_train: ../../data/crop_data/seperate_csv/fold1_train.csv       # The CSV file recording the training samples
  path_csv_val: ../../data/crop_data/seperate_csv/fold1_val_test.csv      # The CSV file recording the validation samples
  format_attn: '.dat'                                                     # Data format of the attention map
model:
  list_downsample_rate: [4, 4, 4, 3, 2]                                   # Down-sampling module architecture in the cropping model
  hidden_activation: Mish                                                 # The hidden activation function in the cropping model (Select from ["LeackyReLU", "Mish"])
  
test:
  checkpoint: /work/kevin8ntust/ViT-Crop-Recognition/AttentionCrop/results/Unet-ch64-4^3*3*2/iteration_100000.pth
  path_csv_test: ./fold1_test_toy.csv                                     # A CSV file recording the testing data
  visualization_downsample_rate: 4                                        # If visualization_downsample_rate is set to 4, the visualization size will be a quarter of the original.
  
  # In the visualization process, we will average the original image and the map. Use these two arguments to set the ratio of the original image
  original_img_ratio_difference_map: 0.4                                  
  original_img_ratio_correctness_map: 0.7
  do_visualization: true                                                  # Do visualization only when this argument is set; otherwise, only show losses and accuracy
  output:
    folder: test_results                                                  # The root folder of the output visualizations
    description: Unet-ch64-4^3*3*2-100000                                 # The sub folder of the output visualizations
```

### Train

```bash
python3 train.py --cfg <The configuration file>
```

### Test

```bash
python3 train.py --cfg <The configuration file>
```

## 3. CroppingModelLoader: Use Cropping Model to Preprocess the Data

We implement the class called `CroppingModelLoader()` in `CroppingModelLoader.py`. Users can use this class to create the dataloader instance, which takes the dataset object as the input and outputs a batch containing the  patches cropped from an input sample.

### Parameters

- **dataset** (`torch.utils.data.Dataset`) - dataset from which to load the data.
- **checkpoint** (`str`) - the path to the cropping model weight.
- **device** (`torch.cuda.device`) - determine which device should the cropping model runs on.
- **max_batch_size** (`int`) - the maximum positive patches we pick from a input image. We will concat these patches in the format BxCxHxW like general PyTorch dataloader outputs.
- **shuffle** (`bool`, optional) – set to `True` to have the data reshuffled at every epoch (default: `False`).
- **positive_sample_threshold** (`float`, optional) - the threshold separating the negative and positive samples. We will only preseve the sample with the predicted attention score greater than the threshold (default: `0.0`).
- **patch_len** (`int`, optional) - the size of the patch size that we want to output (default: `384`).
- **list_downsample_rate** (`list`, optional) - the specification of the down-sampling module in the cropping model (default: `[4, 4, 4, 3, 2]`).
- **hidden_activation** (`str`, optional) - the hidden activation function in the cropping model (default: `"Mish"`).

### Returns

A batch of data in the format BxCxHxW, containing the patches selected by the cropping model.
