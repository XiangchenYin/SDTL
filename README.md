

# [Submited to TMM] Structure-guided Diffusion Transformer for Low-Light Image Enhancement

## Dependencies

```
pip install -r requirements.txt
````

## Download the datasets

Datasets and weights is saved in modelscope:

Modelscope Homepage：

```
https://www.modelscope.cn/models/suxi123/SDTL/files
```

Clone Link：

```
git clone https://oauth2:LisdBebtwskZbrUcnKw8@www.modelscope.cn/suxi123/SDTL.git
```

## How to train?

```
accelerate launch  --multi_gpu --main_process_port 14994 --num_processes 8  --num_machines 1 \
train.py --image_folder  experiments/SDTL_LOLv1 \
--config ./configs/LOLv1.yml --accelerator_train
```

This is also `train.sh`, just run as follows

```
sh train.sh
```

## How to test?

First, you need to move the weights dir "experiments" downloaded from the modelscope community to the project root path, and the dataset dir "data" to the upper level of the root directory;

Test LOLv1 dataset:

```
python evaluate.py --config configs/LOLv1.yml \
--resume experiments/SDTL_LOLv1最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDT_LOLv1
```

Test LOLv2 dataset:

```
python evaluate.py --config configs/LOLv2_real.yml \
--resume  experiments/SDTL_LOLv2_real最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDTL_LOLv2_real 
```

Test LSRW dataset:

```
python evaluate.py --config configs/LSRW.yml \
--resume experiments/SDTL_LSRW最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDTL_LSRW 
```