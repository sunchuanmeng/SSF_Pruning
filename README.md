# SSF_Pruning

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

### 1. VGG-16--CIFAR10

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 93.96%    |[VGG16](https://drive.google.com/file/d/1q_uzAvsAPyQxdaeYWy9NkpnRxwWRr_zc/view?usp=sharing)
| 48.12%    | 93.80%    |[VGG16](https://drive.google.com/file/d/1cNkJm3b2xNJdUFBYSGNM6zgMHksoPQ9f/view?usp=drive_link)|
| 58.40%    | 93.45%    |[VGG16](https://drive.google.com/file/d/1PCQwkbHnY-Rp29VJIKZ--sqOf2MwjTUH/view?usp=drive_link)| 



### 2. ResNet-56--CIFAR10

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 93.26%    |[ResNet-56](https://drive.google.com/file/d/1WE83j7rlKlCp-tslSL6hS-d_mJe4ZQ2r/view?usp=sharing)
| 52.25%    | 93.74%    |[ResNet-56](https://drive.google.com/file/d/1fpox-HIVm80OrBVZ0DtW2B1nodXfWvW5/view?usp=drive_link)|
| 57.11%    | 93.20%    |[ResNet-56](https://drive.google.com/file/d/1CiaVtLlfDeWfbdY34pOx5MuAsHW0EGlm/view?usp=drive_link)|
| 67.04%    | 92.90%    |[ResNet-56](https://drive.google.com/file/d/1F0nfDOvGLsCXMqH8qh_fgc281WjzNsGY/view?usp=drive_link)|
| 69.93%    | 92.56%    |[ResNet-56](https://drive.google.com/file/d/14x5vcZYcy8_qZEQ6hfY4kF-Q1Nky2eu-/view?usp=drive_link)|



### 3. ResNet-110--CIFAR10

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 93.50%    |[ResNet-110](https://drive.google.com/file/d/1YhJHzSBiCsQcNIdamI2_GzclpXvSXcPG/view?usp=sharing)
| 64.69%    | 94.04%    |[ResNet-110](https://drive.google.com/file/d/1YXDlZbK8etUe5POIIezLWRFF-CYuy7iz/view?usp=drive_link)|



### 4. GoogLeNet--CIFAR10

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 95.05%    |[GoogLeNet](https://drive.google.com/file/d/1TXF2OUwkUUWBVAj5Q-QRRO2ZNVRcdmqB/view?usp=sharing)
| 69.73%    | 95.18%    |[GoogLeNet](https://drive.google.com/file/d/1N-wJmTaGmA2ixez5s2sPE7uwF1P4furU/view?usp=drive_link)|



### 5. ResNet-50--ILSVRC2012

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 76.15%    |[ResNet-50](https://drive.google.com/file/d/1H8MlYJCSLmjJOaLjSBMCeh5zfN2bEYT9/view?usp=sharing)
| 57.70%    | 75.14%    |[ResNet-50](https://drive.google.com/file/d/1_KziAUAtXDIokcDcCKcV4m9feO7C1gtq/view?usp=drive_link)| 

### 6. ResNet-18--ILSVRC2012

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 69.66%    |[ResNet-18]( )
| 45.60%    | 67.02%    |[ResNet-18](https://drive.google.com/file/d/1MBUOjkj1actqU7jkuWw6N9jr-MX0LW8x/view?usp=drive_link)| 


### 7. ResNet-56--CIFAR100

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 72.00%    |[ResNet-56](https://drive.google.com/file/d/1EAvfUdbE_8Y569TZRy2KtFoI2B1mFt8E/view?usp=drive_link)
| 56.38%    | 70.78%    |[ResNet-56](https://drive.google.com/file/d/18g8rUa_9fP1VyE1duvx7fc9oi3E9ZjSG/view?usp=drive_link)|

### 8. ResNet-100--CIFAR100

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 72.12%    |[ResNet-110](https://drive.google.com/file/d/1NdKtpQxPPN6fE3g6D5a9y1UnIXh8nckB/view?usp=drive_link)
| 67.42%    | 71.14%    |[ResNet-110](https://drive.google.com/file/d/1HTcngAC0eFEmkHk8Qj9umzGCVxSKP1_y/view?usp=drive_link)|




