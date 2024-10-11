"""
Train and eval functions used in main.py
"""
import torch
import re
import numpy as np

import engine.patch_selection as PS
import engine.image_transmission as IT
from engine.save_image import *

from timm.utils import accuracy
import utils

__all__ = [
    'evaluate', 'attention_histogram'
]

@torch.no_grad()
def evaluate(data_loader, model, server_model, args, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger_masked = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    batch_num = 0
    image_num = 0
    
    if args.output_list:
        output_list = re.findall(r'\d+', args.output_list)
        output_list = list(map(int, output_list))
    else:
        output_list = list(range(50000))
    
    test_logger = utils.MetricLogger(delimiter="  ") # containing token_length, conf_imagenum, min, sum
    acc1_logger = torch.tensor([], device=device)

    for images, target in metric_logger_masked.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]
        
        # compute client model
        with torch.no_grad():
            output, attn, _ = model(images)


        # entropy-aware image transmission
        if args.uncer_mode:
            batch_mask = IT.__dict__[args.uncer_mode](output, args) # B (True -> confident)
        else:
            batch_mask = torch.zeros(output.shape[0], dtype=torch.bool)


        # attention-aware patch selection
        if args.masking_mode:
            len_keep, cams = PS.__dict__[args.masking_mode](attn, args)
        else:
            len_keep, cams = None, ()

        
        # client accuracy update
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


        # compute server model
        with torch.no_grad():
            masked_output = torch.zeros_like(output, device=device)
            masked_output[batch_mask] = output[batch_mask].clone()
            if isinstance(len_keep, torch.Tensor):
                masked_output[~batch_mask] = server_model(images[~batch_mask], len_keep[~batch_mask])
            else:
                for i, batch_mask_ in enumerate(batch_mask):
                    if not batch_mask_:
                        masked_output[[i]] = server_model(images[[i]], len_keep[i].unsqueeze(0))


        # server accuracy update
        acc1_mask, acc5_mask = accuracy(masked_output, target, topk=(1, 5))
        metric_logger_masked.meters['acc1'].update(acc1_mask.item(), n=batch_size)
        metric_logger_masked.meters['acc5'].update(acc5_mask.item(), n=batch_size)


        # logger updates
        test_logger.meters['conf_imagenum'].update(torch.count_nonzero(batch_mask))
        for i, batch_mask_ in enumerate(batch_mask):
            if not batch_mask_:
                test_logger.meters['token_length'].update(len_keep[i].shape[-1])
                if cams:
                    selected_cam = torch.gather(cams[0][i], dim=-1, index=len_keep[i].to(torch.int64))
                    test_logger.meters['min'].update(selected_cam.min())
                    test_logger.meters['sum'].update(selected_cam.sum())



        # save images
        if args.output_dir:
            acc1_logger = save_acc1(output, masked_output, target, image_num, args, acc1_logger)
            image_num = save_image(images, len_keep, cams, image_num, output_list, args)


        batch_num += 1



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger_masked.synchronize_between_processes()
    test_logger.synchronize_between_processes()
    client_acc = metric_logger.meters['acc1'].global_avg
    server_acc = metric_logger_masked.meters['acc1'].global_avg
    conf_imagenum = test_logger.meters['conf_imagenum'].sum
    token_length = test_logger.meters['token_length'].global_avg
    if test_logger.meters.get('min'):
        min_cam = test_logger.meters['min'].global_avg
        cam_sum = test_logger.meters['sum'].global_avg
    else:
        min_cam, cam_sum = None, None
    
    # print results to terminal
    print('* Masking mode:: {}, {} / Confidence criterion:: {}, {}'.format( \
     args.masking_mode, args.masking_th, args.uncer_mode, args.uncer_th))
    print(f"* Sent token number:: {token_length:.4f} " \
        + f"/ Averaged minimum attention:: {min_cam:.4f} / Averaged sum of attention:: {cam_sum:.4f}")
    print('* Total confident image:: {}'.format(conf_imagenum))
    comm_cost = (50000-conf_imagenum)/50000 * token_length / 196
    print('* Communication cost:: {}'.format(comm_cost))
    print(f'Accuracy of the client model: {client_acc:.2f} %')
    print(f'Accuracy of the server model: {server_acc:.2f} %')


    # print results to text file
    if args.output_dir:
        acc1_logger_list = [torch.zeros_like(acc1_logger, device=device) for _ in range(args.world_size)]
        torch.distributed.all_gather(acc1_logger_list, acc1_logger)
        if utils.get_rank() == 0:
            acc1_logger_list = torch.cat(acc1_logger_list, dim=0)
            acc1_logger_list = acc1_logger_list[acc1_logger_list[:,0].sort().indices].cpu().numpy()
            p = Path(args.output_dir) / 'acc1.txt'
            np.savetxt(p, acc1_logger_list, fmt=r'%d', delimiter=', ')
            

    if utils.get_rank() == 0:
        p = Path('./accuracy.txt')
        initline = False if p.exists() else True
        with p.open(mode='a',encoding='UTF-8') as f:
            if initline:
                f.write('masking_mode, masking_th, patch_num, uncer_mode, uncer_th, conf_img, comm_cost, accuracy\n')
            f.write(f'{args.masking_mode}, {args.masking_th}, {token_length:.2f}, ' \
                + f'{args.uncer_mode}, {args.uncer_th:.2f}, {conf_imagenum}, ' \
                + f'{comm_cost:.4f}, {server_acc:.4f}\n')
            


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, {k: meter.global_avg for k, meter in metric_logger_masked.meters.items()}



@torch.no_grad()
def attention_histogram(data_loader, model, args, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    batch_num = 0
    hist_LB = 0
    hist_UB = 0.01
    num_bins = 100
    attn_hist = torch.zeros(num_bins, device=device)
    
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute client model
        with torch.no_grad():
            output, attn, _ = model(images)

        # get attention score
        attention = PS.attention_processing(attn, args)

        # add into histogram
        attn_hist += torch.histc(attention.flatten(), bins=num_bins, min=hist_LB, max=hist_UB).to(device)
                


        # accuracy update
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)



        batch_num += 1



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    client_acc = metric_logger.meters['acc1'].global_avg
    torch.distributed.all_reduce(attn_hist)
    
    print(f'Accuracy of the model: {client_acc:.2f}')

    # print histogram
    if args.output_dir:
        bins = torch.linspace(hist_LB, hist_UB, num_bins+1)[1:]
        attn_hist = attn_hist.cpu()
        attn_hist_norm = attn_hist.double() / (50000 * 196) # # of samples & patches: 50000, 196
        plt.figure()
        plt.plot(bins, attn_hist_norm)
        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / 'attn_hist_norm.png')

        txt = torch.stack((bins,attn_hist,attn_hist_norm), dim=0).numpy()
        np.savetxt(Path(args.output_dir) / 'attn_hist.txt', txt, delimiter=', ')
    

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
