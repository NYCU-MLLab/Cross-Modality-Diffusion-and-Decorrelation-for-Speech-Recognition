import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



# def criteria_barlow_twins(z1,z2,scale_loss=1/32,lambd=3.9e-3):
#     # print("z1",z1.t())
#     # print("z2", z2)
#     losses = torch.empty(z1.size(dim = 0)).to(device = z1.device)
    
#     z11, z22 = z1, z2
#     for i, (z1, z2) in enumerate(zip(z11, z22)):
#         z1 = z1.unsqueeze(0)
#         z2 = z2.unsqueeze(0)
#         # print("z1", z1)
#         # print("z2", z2)
#         c = z1.t() @ z2
#         # print("c",c)
#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(scale_loss)
#         off_diag = off_diagonal(c).pow_(2).sum().mul(scale_loss)

#         loss = on_diag + lambd * off_diag
    
#         losses[i] = loss
    
#     return losses

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten() 


def criteria_barlow_twins(z1,z2,scale_loss=1/32,lambd=3.9e-3):
    z1 = (z1 - z1.mean(0)) / z1.std(0)
    z2 = (z2 - z2.mean(0)) / z2.std(0)
    c = z1.t() @ z2
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(scale_loss)
    off_diag = off_diagonal(c).pow_(2).sum().mul(scale_loss)
    loss = on_diag + lambd * off_diag    
    return loss

# Reference
# https://colab.research.google.com/drive/1hYHb0FTdKQCXZs3qCwVZnSuVGrZU2Z1w?usp=sharing#scrollTo=7MQnmwsWi6lc
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4

## This can be modified!!!!
class Projection(nn.Module):
    def __init__(self, d_in, d_out, p=0.5):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        # embeds = embed1 + embed2
        return embeds


class MyEnsemble(nn.Module):
    def __init__(self, d_in_1, d_out_1, d_in_2, d_out_2):
        super(MyEnsemble, self).__init__()
        self.modelA = Projection(d_in_1, d_out_1)
        self.modelB = Projection(d_in_2, d_out_2)
        
    def forward(self, x1, x2):
        # print("x11",x1)
        x1 = self.modelA(x1)
        # print("x12",x1)
        x2 = self.modelB(x2)

        return x1, x2