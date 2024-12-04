import torch
import numpy as np
import utils
import os
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)
                x_output = x_output[:, :, :h, :w]
                
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}"))
                print(f"processing image {y[0]}")

    def restore_slid(self, path):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        
        img_list = [os.path.join(path, i) for i in os.listdir(path)]

        with torch.no_grad():
            
            for i, img_name in enumerate(img_list):
                y = img_name.split('/')[-1]

                img = cv2.imread(img_name)
                height, width, channels = img.shape
                col, row = 4, 4
                crop_size = (height//col, width//row)
                patch_img = slide_transform(img, crop_size)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                result = torch.ones(1, 3, height, width).float()
                for i in range(0, col):
                    for j in range(0, row):
                        
                        x_cond = (transform(patch_img[i, j, :, :, :])/255.0).unsqueeze(0).float().to(self.diffusion.model.device)
                        b, c, h, w = x_cond.shape
                        img_h_32 = int(32 * np.ceil(h / 32.0))
                        img_w_32 = int(32 * np.ceil(w / 32.0))
                        # condt = x_cond.cpu().squeeze(0).permute(1,2,0).numpy()
                        # cv2.imwrite('condt.png', condt) 

                        x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                        x_output = self.diffusive_restoration(x_cond)
                        x_output = x_output[:, :, :h, :w]
                        result[:, :, i*crop_size[0]:i*crop_size[0]+crop_size[0], j*crop_size[1]:j*crop_size[1]+crop_size[1]] = x_output

                utils.logging.save_image(result, os.path.join(image_folder, f"{y}"))
                print(f"processing image {y}")
                torch.cuda.empty_cache() 

    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]

    def restore1(self, path):
        image_folder = os.path.join(self.args.image_folder)
        
        img_list = [os.path.join(path, i) for i in os.listdir(path)]

        with torch.no_grad():
            transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            
            for i, img_name in enumerate(img_list):
                y = img_name.split('/')[-1]
                # if os.path.exists(os.path.join(image_folder, f"{y}")):
                #     continue
                img = cv2.imread(img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channels = img.shape

                x_cond = (transform(img)).unsqueeze(0).float().to(self.diffusion.model.device)

                import torch.nn.functional as F

                # 假设 x 是形状为 (batch_size, channels, height, width) 的 4D Tensor
                # x_cond = F.interpolate(x_cond, scale_factor=0.5, mode='bilinear', align_corners=False)
                
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)
                x_output = x_output[:, :, :h, :w]
                # x_output = F.interpolate(x_output, scale_factor=2, mode='bilinear', align_corners=False)

                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}"))
                print(f"processing image {y}")
                torch.cuda.empty_cache() 
        print('Finished!!!!')
def slide_transform(img, crop_size):
    h,w,c = img.shape

    patch_col = h//crop_size[0]
    patch_row = w//crop_size[1]

    patch_img = np.ones((patch_col, patch_row, crop_size[0], crop_size[1], 3))

    for i in range(patch_col):
        for j in range(patch_row):

            patch_img[i, j, :, :, :] = img[i*crop_size[0]:i*crop_size[0]+crop_size[0], j*crop_size[1]:j*crop_size[1]+crop_size[1], :]

    # 创建一个新的figure，并设置子图布局为2行4列
    # fig, axs = plt.subplots(nrows=patch_col, ncols=patch_row, figsize=(12, 6))
    # 可选：调整子图间距

    # for i in range(patch_col):
    #     for j in range(patch_row):
    #         axs[i, j].imshow(patch_img[i,j,:,:,::-1].astype(np.uint8))

    # plt.tight_layout()

    # # 显示图像
    # plt.show()
    return patch_img

# img = cv2.imread('1.png')
# slide_transform(img, 2,4)


