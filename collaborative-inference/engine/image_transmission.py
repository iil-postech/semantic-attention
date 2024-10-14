"""
Selective image transmission functions used in engine.py

Input: client prediction output (tensor: BxLb), args

Output: batch mask (True is considered as confident -- do not re-infer)
"""
import torch
from torch.nn.functional import softmax

__all__ = [
    'shannon_entropy', 'min_entropy', 'margin'
]


@torch.no_grad()
def shannon_entropy(output, args):
    sf = softmax(output, dim=-1)
    uncertainty = torch.sum(- torch.mul(sf, sf.log2()), dim=-1)
    return uncertainty < args.uncer_th


@torch.no_grad()
def min_entropy(output, args):
    sf = softmax(output, dim=-1)
    uncertainty = - sf.max(dim=-1).values.log2()
    return uncertainty < args.uncer_th


@torch.no_grad()
def margin(output, args):
    sf = softmax(output, dim=-1)
    top_sf = sf.topk(k=2, dim=1).values
    confidence = top_sf[:,0] - top_sf[:,1]
    return confidence > args.uncer_th