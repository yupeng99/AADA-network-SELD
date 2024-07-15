# ADA-network-SELD

This repository is an official PyTorch implementation of the paper "Automated Audio Data Augmentation Network Using Bi-Level Optimization for Sound Event Localization and Detection"

------



## 1. Generate the data_list file

```python
 python create_list.py
```

After running the program, it will generate the paths for all the audio files in the corresponding dataset. When loading the audio data later, it will search for the detailed paths from that file. Please refer to the following illustration:

![image-20240715163045209](img/image-20240715163045209.png)

------

## 2. Configure Parameters in dcase2023.yaml file

**Common Configuration**：

- dataset_root: The path of the generated file after Step 1.
- dataset_list_train: Using the generated training dataset file
- dataset_list_valid: Using the generated test dataset file
- use_aug: Whether to use data augmentation (not applicable to AADA policy)

**The configuration changes required when using different models are as follows:**：

- model: Model name
- model_features_transform: Feature extraction method
- batch_size
- input_shape: Input model dimension
- output_shape : Output model dimension
- dataset_chunk_size_seconds: Audio Segment Length (in seconds)

## 3. Training

```python
python -c ./configs/dcase2023.yaml
```

------

## 4. Pretraining

The uploaded files include the Baseline model pre-trained. 

Due to the size limitation, the pre-model for CRNN10 is located at [here](https://pan.baidu.com/s/13NMMFdtfMRo04pcJuUN_sw)

, and the pre-model for Resnet-Conformer is located at [here](链接：https://pan.baidu.com/s/19OcEG02on3ROj6gRoBhrog 提取码：aada).

