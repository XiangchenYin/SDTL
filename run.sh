python evaluate.py --config configs/LOLv1.yml \
--resume experiments/SDTL_LOLv1最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDT_LOLv1


python evaluate.py --config configs/LOLv2_real.yml \
--resume  experiments/SDTL_LOLv2_real最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDTL_LOLv2_real 


python evaluate.py --config configs/LSRW.yml \
--resume experiments/SDTL_LSRW最终结果/model_best_epoch.pth.tar \
--image_folder  result/SDTL_LSRW 





