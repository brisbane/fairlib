from typing import List, Mapping, Optional

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

Outputs = Mapping[str, List[torch.Tensor]]


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class DiffLoss(torch.nn.Module):
    """
    compute the Frobenius norm of two tensors
    """
    # From: Domain Separation Networks (https://arxiv.org/abs/1608.06019)
    # Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, D1, D2):
        D1=D1.view(D1.size(0), -1)
        D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-6)

        D2=D2.view(D2.size(0), -1)
        D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
        D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-6)

        mean = torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
#        print ("This is the mean:", mean)
        return mean

class CorrLoss(torch.nn.Module):
    """
    Compute the Pearson correlation coefficient
    """
    #def __init__(self , diagonal=True):
    def __init__(self ):
       super(CorrLoss, self).__init__()
       #self.diagonal = diagonal
    #D1 INPUTS
    #D2 TARGETS

    def forward(self, D1, D2):
#       print (D1.shape)
 #      print (D1)
       D1=D1.view(D1.size(0), -1)
  #     print (D1)
   #    print (D2)
       D1_norm=torch.norm(D1, p=2, dim=1, keepdim=True).detach()
       D1_norm=D1.div(D1_norm.expand_as(D1) + 1e-9)
    #   print (D1.shape)
     #  print (D1_norm.shape)
       D2=D2.view(D2.size(0), -1)
       D2_norm=torch.norm(D2, p=2, dim=1, keepdim=True).detach()
       D2_norm=D2.div(D2_norm.expand_as(D2) + 1e-9)
       cos = nn.CosineSimilarity(dim=1, eps=1e-9)

       pearson2 = torch.square(cos(D1_norm - D1_norm.mean(dim=1, keepdim=True), D2_norm - D2_norm.mean(dim=1, keepdim=True)))
       #print ("pre matrix:", pearson2)
       corrmean = pearson2.sum()/pearson2.numel()
#       print ("This is the pearson:", corrmean)
#       return F.cosine_similarity(D1_norm, D2_norm).pow(2).mean()
       return corrmean
