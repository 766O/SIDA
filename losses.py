import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def one_hot(index, classes):
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    mask = torch.Tensor(size).fill_(0).cuda()
    index = index.view(view)
    ones = 1.
    return mask.scatter_(1, index, ones)

class UncCELoss(nn.Module):
    def __init__(self, num_classes=19, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=255, weight=None):
        super(UncCELoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.classs = num_classes
        self.size_average = size_average
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.ignore = ignore
        self.weights = weight
        self.raw = False
        if (num_classes < 19):
            self.raw = True

    def forward(self, input, target, conf, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C) # torch.Size([589824, 19])
        target = target.view(-1) # torch.Size([589824]) (768*768)
        conf = conf.view(-1)
        #import pdb;pdb.set_trace()
        
        if self.ignore is not None:
            valid = (target != self.ignore) # ignore 값 필터링
            input = input[valid]
            target = target[valid]
            conf = conf[valid]

        if self.one_hot:
            target_onehot = one_hot(target, input.size(1))

        probs = F.log_softmax(input, dim=1)
        probs = (probs * target_onehot) # CE loss -> Sofxmax * target_one_hot torch.Size([425031, 19])
        
        
        probs = torch.sum(probs, dim=1) # torch.Size([425031])
        
        probs = conf * probs + probs
        #probs = torch.exp(conf) * probs 
       
        batch_loss = -probs    
        #probs = torch.exp(conf-0.05) * probs 
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
            
        return loss
    
    def forward_backup(self, input, target, conf, eps=1e-5):
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C) # torch.Size([589824, 19])
        target = target.view(-1) # torch.Size([589824]) (768*768)
        conf = conf.view(-1)
        
        
        if self.ignore is not None:
            valid = (target != self.ignore) 
            input = input[valid]
            target = target[valid]
            conf = conf[valid]

        if self.one_hot:
            target_onehot = one_hot(target, input.size(1))

        probs = F.log_softmax(input, dim=1)
        probs = (probs * target_onehot) # CE loss -> Sofxmax * target_one_hot torch.Size([425031, 19])
        
        
        probs = torch.sum(probs, dim=1) # torch.Size([425031])
        
        probs = torch.exp(conf) * probs 
       
        batch_loss = -probs    
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
            
        return loss