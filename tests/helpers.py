"""Tensor-aware equality extracted from the reviewed diagnostic helper."""
import torch

def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
