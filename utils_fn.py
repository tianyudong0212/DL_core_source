import torch
import torch.nn as nn

def contrastive_loss(q, k0, neg, temperature=0.07):
    '''calculate the contrastive-like loss in a batch
    
    Args:
        q(size: [h_dim]): one representation of psg_i 
        k0(size: [h_dim]): the other repr of psg_i, used as positive sample
        neg(size: [bsz-1, h_dim]): the negative samples, namely repr of k_1, k_2, ..., k_(bsz-1)
        temperature: the temperature coefficient
    '''
    upper = torch.exp(q @ k0)
    neg_dot = q @ neg.T
    lower = sum([torch.exp(neg_dot[i]) for i in range(neg_dot.shape[0])])
    return - torch.log((upper/temperature) / (lower/temperature))
