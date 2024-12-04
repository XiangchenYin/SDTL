

# [Submited to TMM] Structure-guided Diffusion Transformer for Low-Light Image Enhancement

## Dependencies

```
pip install -r requirements.txt
````

## Download the datasets

我们将数据集和模型的权重保存在了modelscope社区：

主页：

```
https://www.modelscope.cn/models/suxi123/SDTL/files
```

clone链接：

```
git clone https://oauth2:LisdBebtwskZbrUcnKw8@www.modelscope.cn/suxi123/SDTL.git
```

## How to train?

```
accelerate launch  --multi_gpu --main_process_port 14994 --num_processes 8  --num_machines 1 \
train.py --image_folder  experiments/SDTL_LOLv1 \
--config ./configs/LOLv1.yml --accelerator_train
```

这同时也是`train.sh`的内容，或者可以

```
sh train.sh
```

## How to test?

首先需要将从modelscope社区下载的权重移到项目根路径；

测试LOLv1数据集：

```
python evaluate.py --config configs/LOLv1.yml \
--resume experiments/SDTL_LOLv1最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDT_LOLv1
```

测试LOLv2数据集：

```
python evaluate.py --config configs/LOLv2_real.yml \
--resume  experiments/SDTL_LOLv2_real最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDTL_LOLv2_real 
```

测试LSRW数据集：

```
python evaluate.py --config configs/LSRW.yml \
--resume experiments/SDTL_LSRW最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDTL_LSRW 
```