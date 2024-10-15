import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path


from datasets import build_dataset
from engine import *


import models_client #models_client.py
import models_server #models_server.py
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT evaluation script', add_help=False)
 
    # Masking parameters
    parser.add_argument('--masking_mode', default='topk', choices=['topk', 'attention_threshold', 'attention_sum_threshold', 'random'],type=str, help='image masking mode')
    parser.add_argument('--attention_mode', default='mean', choices=['mean', 'rollout'], type=str, help='attention score computing mode')
    parser.add_argument('--masking_th', default=0.97, type=float, help='masking threshold')


    # Uncertainty system parameters
    parser.add_argument('--uncer_mode', default='', choices=['','shannon_entropy','margin','min_entropy'], type=str, help='uncertainty measure')
    parser.add_argument('--uncer_th', default=0.8, type=float)

    parser.add_argument('--attention_histogram', action='store_true')
    parser.set_defaults(attention_histogram=False)

    # Masking color
    parser.add_argument('--cmap', default= 'RdPu', choices=['RdPu', 'rainbow', 'PuBu'], type=str, help='attention heatmap color')
    
    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--server-model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of server model')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--use-script', action='store_true')
    parser.set_defaults(use_script=False)


    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
            
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_list', default='', type=str, help='a list of images to save (example: 1,123,15300), empty for full-saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--visible_devices', default='', type=str)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)


    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_val, _ = build_dataset(args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")

    model = models_client.__dict__[args.model](pretrained = True)          
                                
    model.to(device)
    model.eval()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    for p in model.parameters():
        p.requires_grad = False
    
                     
    if not args.attention_histogram:
        print(f"Creating server model: {args.server_model}")
        
        server_model = models_server.__dict__[args.server_model](pretrained = True)          

        server_model.to(device)
        server_model.eval()
        if args.distributed:
            server_model = torch.nn.parallel.DistributedDataParallel(server_model, device_ids=[args.gpu])
        for p in server_model.parameters():
            p.requires_grad = False


    # Validation
    for _ in utils.init_script(args):
        
        if not args.attention_histogram:
            client_test_stats, server_masked_stats = evaluate(data_loader_val, model, server_model, args, device)
        else:
            client_test_stats = attention_histogram(data_loader_val, model, args, device)




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        p = Path(args.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / 'original').mkdir(parents=True, exist_ok=True)
        if args.masking_mode:
            (p / 'masked').mkdir(parents=True, exist_ok=True)
        if args.masking_mode in ['topk', 'attention_threshold', 'attention_sum_threshold']:
            (p / 'heatmap').mkdir(parents=True, exist_ok=True)
    main(args)
