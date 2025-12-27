import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Log_Cosin_Loss(nn.Module):
    def forward(self, x, y):
        x = F.normalize(x.float())
        y = F.normalize(y.float())
        cos_sim_metrix = torch.mm(x, y.T)
        mask = torch.eye(x.size(0), x.size(0)).bool().cuda()
        cos_sim = (cos_sim_metrix*mask).sum(dim=1)
        loss = -torch.log((cos_sim+1)/2)
        return loss.mean()

class Cosin_Loss(nn.Module):
    def forward(self, x, y):
        x = F.normalize(x.float())
        y = F.normalize(y.float())
        cos_sim_metrix = torch.mm(x, y.T)
        mask = torch.eye(x.size(0), x.size(0)).bool().cuda()
        loss = 1-(cos_sim_metrix*mask).sum(dim=1)
        return loss.mean()

class Relative_Cosin_Loss(nn.Module):
    def forward(self, x, y):
        x = F.normalize(x.float())
        y = F.normalize(y.float())
        cos_sim_metrix = torch.mm(x, y.T)
        cos_sim_metrix = F.log_softmax(cos_sim_metrix, dim=1)
        mask = torch.eye(x.size(0), x.size(0)).bool().cuda()
        cos_sim = (cos_sim_metrix*mask).sum(dim=1)
        return (1-cos_sim).mean()

class Smoth_CE_Loss(nn.Module):
    def __init__(self, ls_=0.9):
        super().__init__() 
        self.ls_ = ls_

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=1)
        loss = -logprobs * (target*self.ls_)
        loss = loss.sum(dim=1)
        return loss

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=1)
        loss = -logprobs * target
        loss = loss.sum(dim=1)
        return loss

class DenseCrossEntropy_Multi_Label(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.index = [0, 2, 5, 8, 10, 12, 14, 17, 19, 21, 26, 28, 30]

    def forward(self, pred, label):
        for i in range(len(self.index)-1):
            temp = pred[:, self.index[i]:self.index[i+1]].softmax(dim=1)
            pred[:, self.index[i]:self.index[i+1]] = temp
        loss = -pred.log() * label

        loss = loss.sum(dim=1)
        loss = loss.mean()
        return loss

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.3, crit="multi_ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if crit == "ce":
            self.crit = DenseCrossEntropy()
        elif crit == "multi_ce":
            self.crit = DenseCrossEntropy_Multi_Label()
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.reduction = reduction
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # labels = F.one_hot(labels.long(), logits.shape[-1]).float()
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, max_m=0.8, num_calsses=30, label_index=[0, 2, 5, 8, 10, 12, 14, 17, 19, 21, 26, 28], crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if crit == "ce":
            self.crit = DenseCrossEntropy()
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.m = m
        self.max_m = max_m
        self.label_index = label_index
        self.label_index.append(num_calsses)
        self.reduction = reduction
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def update(self, c_epoch, num_epochs):
        self.m = self.m+(self.max_m-self.m)*((num_epochs-c_epoch)/num_epochs)
        self.last_epoch = c_epoch
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, logits, labels):
        logits = logits.float()
        losses = 0
        for i in range(len(self.label_index)-1):
            cosine = logits[:, i, :]
            mask = torch.zeros_like(labels)
            mask[self.label_index[i]:self.label_index[i+1]] = 1
            label = labels*mask
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = phi.type(cosine.type())
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # label = F.one_hot(label.long(), logits.shape[-1]).float()
            output = (label * phi) + ((1.0 - label) * cosine)
            output *= self.s
            loss = self.crit(output, label)
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()
            losses += loss
        return losses