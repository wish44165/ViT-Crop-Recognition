# Cropping Model

This folder is related to the cropping model, including the following features:
1. The program generates ground truth to train the cropping model
2. The training program

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

## 2. Training Program

Still in progress...
