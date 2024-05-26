# EXP_Pruning
EXP way : Complete pruning using the expectation scaling factor

## Running Code

    In this code, you can run our models on CIFAR-10 and ImageNet dataset. The code has been tested by Python 3.6, Pytorch 1.6 and CUDA 10.2 on Windows 10.
    For the channel mask generation, no additional settings are required. You can just set the required parameters in main.py and it will run.

## parser
```shell
&&& main.py &&&
/data_dir/ : Dataset storage address
/dataset/ ： dataset - CIFAR10 or Imagenet
/lr/ ： initial learning rate
/lr_decay_step/ ： learning rate decay step
/resume/ ： load the model from the specified checkpoint
/resume_mask/ ： After the program is interrupted, the task file can be used to continue running
/job_dir/ ： The directory where the summaries will be stored
/epochs/ ： The num of epochs to fine-tune
/start_cov/ ： The num of conv to start prune
/compress_rate/ ： compress rate of each conv
/arch/ ： The architecture to prune


&&& cal_flops_params.py &&&
/input_image_size/ : 32(CIFAR-10) or 224(ImageNet)
/arch/ ： The architecture to prune
/compress_rate/ ： compress rate of each conv
```

## Model Training

For the ease of reproducibility. we provide some of the experimental results and the corresponding pruned rate of every layer as belows:

### 1. VGG-16

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 93.96%    |[VGG16](https://drive.google.com/file/d/1q_uzAvsAPyQxdaeYWy9NkpnRxwWRr_zc/view?usp=sharing)
| 53.1%     | 93.64%    |[A](https://drive.google.com/file/d/1S4he_cv9NGbtT3HL13uQ5qZQ5_r_3W9N/view?usp=sharing)|
| 53.1%     | 93.55%    |[B](https://drive.google.com/file/d/1Df7LM3kNULiqhT97TXgAlcvqETcJXwzK/view?usp=sharing)| 
| 60.8%     | 93.50%    |[A](https://drive.google.com/file/d/1qs1cFQBko9HdNno7XeybT7xVTPH-hAGl/view?usp=sharing)| 


### 2. ResNet-56

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 93.26%    |[ResNet-56](https://drive.google.com/file/d/1WE83j7rlKlCp-tslSL6hS-d_mJe4ZQ2r/view?usp=sharing)
| 53.5%     | 93.54%    |[A](https://drive.google.com/file/d/1WhW7O0-GDvZCLpwvXdCLVWK5kddgk94z/view?usp=sharing)|
| 53.5%     | 93.82%    |[B](https://drive.google.com/file/d/1qs1cFQBko9HdNno7XeybT7xVTPH-hAGl/view?usp=sharing)|
| 60.5%     | 93.48%    |[A](https://drive.google.com/file/d/1qs1cFQBko9HdNno7XeybT7xVTPH-hAGl/view?usp=sharing)|


### 3. ResNet-110

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 93.50%    |[ResNet-110](https://drive.google.com/file/d/1YhJHzSBiCsQcNIdamI2_GzclpXvSXcPG/view?usp=sharing)
| 60.1%     | 93.96%    |[A](https://drive.google.com/file/d/1qTeTYPiyVZCPaEhzH1z_HvDyKlWuQtoF/view?usp=sharing)|
| 60.1%     | 93.73%    |[B](https://drive.google.com/file/d/1UNPm5DWO8JYZGtbWAmELVkjb5UDcamem/view?usp=sharing)|


### 4. GoogLeNet

| Flops     | Accuracy  |way and Model                | Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|-----------|-----------|-----------------------------|
| 100%      | 95.05%    |[GoogLeNet](https://drive.google.com/file/d/1TXF2OUwkUUWBVAj5Q-QRRO2ZNVRcdmqB/view?usp=sharing)
| 62.1%     | 95.02%    |[A](https://drive.google.com/file/d/19N_maLGWQAlO4m_S77Qm4m791oMoe4ha/view?usp=sharing)|
| 62.1%     | 94.95%    |[B](https://drive.google.com/file/d/1woyidXT9O-TQHiieEUrSu7UTbxVPkvtA/view?usp=sharing)|


### 5. ResNet-50

| Flops     | Accuracy  |way and Model                | Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|-----------|-----------|-----------------------------|
| 100%      | 76.15%    |[ResNet-50](https://drive.google.com/file/d/1H8MlYJCSLmjJOaLjSBMCeh5zfN2bEYT9/view?usp=sharing)
| 53.05%    |  75.71%    |[A](https://drive.google.com/file/d/1qZsJibWGkZTp6AiVOt_OrLZz-_crKYEo/view?usp=sharing)| 60.63%     | 74.53%    |[A](https://drive.google.com/file/d/1A9JiEkOXTKbezOscs5_crf3rvio5HSIz/view?usp=sharing)
| 53.05%    |  75.76%    |[B](https://drive.google.com/file/d/12J-HEY1CMqREsfQEMNfiYpw7ON90WuF8/view?usp=sharing)| 60.00%     | 75.02%    |[B](https://drive.google.com/file/d/1kEAO46J2j5k6wnMeh9dKh-EvdaDd0M6C/view?usp=sharing)






