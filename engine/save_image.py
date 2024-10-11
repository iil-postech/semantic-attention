"""
Image save function used in engine.py

Input: image (tensor: BxCxHxW), len_keep (tuple or tensor: Bxn), cams (tuple: ?xBxN),
    file number (int), output list (list), args

"""
import torch
import torch.nn.functional as F
from torchvision.utils import save_image as save_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pathlib import Path
import matplotlib.pyplot as plt



@torch.no_grad()
def save_image(images, len_keep, cams, image_num, output_list, args):

    if len(cams) != 0:
        _, _, height, width = images.shape
        pixels = height * width
        num_patches = cams[0].shape[-1]
        patch_size = int((pixels // num_patches) ** 0.5)

    
    orig_dir = Path(args.output_dir) / 'original'
    mask_dir = Path(args.output_dir) / 'masked'
    heat_dir = Path(args.output_dir) / 'heatmap'
        
    for img_idx, image in enumerate(images):

        file_num = args.world_size * image_num + args.rank
        if file_num in output_list:

            ## 1. save original image
            x = image.clone().squeeze()
            x = torch.stack([x_*std_+mean_ for x_,std_,mean_ in zip(x,IMAGENET_DEFAULT_STD,IMAGENET_DEFAULT_MEAN)],dim=0)
            save_(x, orig_dir / f'{file_num}.png')

            ## 2. save masked image
            if len_keep is not None:
                ones = torch.ones(len_keep[img_idx].shape[-1], device=image.device)
                mask = torch.zeros(num_patches, device=image.device).scatter_(dim=0, index=len_keep[img_idx].type(torch.int64), src=ones) # 196
                mask = mask.reshape(1,1,height//patch_size,width//patch_size) # 1 x 1 x 14 x 14
                mask = F.interpolate(input=mask, scale_factor=patch_size, mode='nearest').squeeze() # 224 x 224

                mask[mask == 0] = 0.2
                x = torch.cat((x,mask.unsqueeze(0)),dim=0)
                save_(x, mask_dir / f'masked_{file_num}_{len_keep[img_idx].shape[0]}.png')

            ## 3. save class activation heatmap
            for i, cam in enumerate(cams):
                cam_ = cam[img_idx].reshape(height//patch_size,width//patch_size)
                plt.rcParams["figure.autolayout"] = True
                fig, ax = plt.subplots()
                plt.imshow(cam_.cpu(), cmap=args.cmap, interpolation='nearest')
                plt.axis('off')
                plt.savefig(heat_dir / f'heatmap{i+1}_{file_num}.png', bbox_inches='tight')
                plt.cla()
                plt.clf()
                plt.close()


        image_num += 1

    return image_num
    


@torch.no_grad()
def save_acc1(client_output, server_output, target, image_num, args, logger):
    numbers = args.world_size * (torch.arange(target.shape[0], device=client_output.device).int() + image_num) + args.rank
    c_idx = client_output.argmax(dim=1).to(target.device).eq(target).int()
    s_idx = server_output.argmax(dim=1).to(target.device).eq(target).int()
    log_ = torch.cat((numbers.unsqueeze(1),c_idx.unsqueeze(1),s_idx.unsqueeze(1)), dim=1)
    logger = torch.cat((logger,log_), dim=0)
    return logger
        