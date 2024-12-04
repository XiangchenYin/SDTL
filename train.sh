accelerate launch  --multi_gpu --main_process_port 14994 --num_processes 8  --num_machines 1 \
train.py --image_folder  experiments/SDTL_LOLv1 \
--config ./configs/LOLv1.yml --accelerator_train

