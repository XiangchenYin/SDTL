

accelerate launch \
 --multi_gpu --main_process_port 14994 --num_processes 4  --num_machines 1 \
train.py --image_folder  experiments/SDTL_LOLv2 \
--config ./configs/LOLv2_real.yml --accelerator_train 2>&1  | tee "LOLv2_real.log"



