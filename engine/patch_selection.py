"""
Patch selection functions used in engine.py

Input: attn (tensor: LxBxHxNxN), args

Output: len_keep (tuple or tensor: Bxn), heatmap (tuple of tensor: ?xBxN)
"""
import torch
import re

__all__ = [
    'random', 'topk', 'attention_threshold', 'attention_sum_threshold', 'attention_processing'
]


def compute_rollout_attention(all_layer_matrices, start_layer=0): # L x (B x H x) N x N
    # adding residual consideration
    num_tokens = all_layer_matrices.shape[-1]
    all_layer_matrices += torch.eye(num_tokens, device=all_layer_matrices.device).repeat(*all_layer_matrices.shape[:-2],1,1)
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = torch.matmul(all_layer_matrices[i], joint_attention)
    return joint_attention # (B x H x) N x N

def sel_topk(cam, th):
    return torch.topk(cam, int(th)).indices.to(cam.device)

def sel_th(cam, th):
    len_keep = []
    index = torch.arange(cam.shape[-1], device=cam.device)
    for cam_ in cam:
        len_keep.append(index[cam_ >= th].clone())
    return len_keep

def sel_sum_th(cam, th):
    len_keep = []
    cam_sort, index = cam.sort(dim=-1, descending=True)
    cam_sort.to(cam.device).cumsum_(dim=-1)
    index.to(cam.device)
    for cam_sort_, index_ in zip(cam_sort, index):
        len_keep.append(index_[cam_sort_ <= th].clone())
    return len_keep

def normalize(cam):
    if isinstance(cam, torch.Tensor):
        cam_norm = cam / cam.sum(dim=-1, keepdim=True)
        return cam_norm.nan_to_num_()
    elif isinstance(cam, tuple):
        cam_norm = []
        for cam_ in cam:
            cam_norm_ = cam_ / cam_.sum(dim=-1, keepdim=True)
            cam_norm.append(cam_norm_.nan_to_num_())
        return cam_norm



@torch.no_grad()
def attention_processing(attn, args):
    '''
    Input: attn (tensor: LxBxHxNxN), args
    Output: attn (tensor: BxN)
    '''
    if args.attention_mode == 'mean':
        attn = attn[-1]
    elif args.attention_mode == 'rollout':
        attn = compute_rollout_attention(attn, start_layer=0)
    attn = attn.mean(dim=1)[:,0,1:]
    attn = normalize(attn)
    return attn


@torch.no_grad()
def random(attn, args):
    ones = torch.ones((attn.shape[1], attn.shape[-1]-1), dtype=torch.float, device=attn.device)
    len_keep = torch.multinomial(ones, int(args.masking_th)).to(ones.device)
    return len_keep, ()


@torch.no_grad()
def topk(attn, args):
    attn = attention_processing(attn, args)
    len_keep = sel_topk(attn, args.masking_th)
    return len_keep, (attn, )


@torch.no_grad()
def attention_threshold(attn, args):
    attn = attention_processing(attn, args)
    len_keep = sel_th(attn, args.masking_th)
    return len_keep, (attn, )


@torch.no_grad()
def attention_sum_threshold(attn, args):
    attn = attention_processing(attn, args)
    len_keep = sel_sum_th(attn, args.masking_th)
    return len_keep, (attn, )