import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def precision(output, target):
    pred = output.argmax(1)
    prec = 0
    for i in range(output.shape[-1]):
        p = pred==i
        g = target==i
        prec += (p*g).sum()/(p.sum()+1e-7)
    return prec.mul_(100.0 / output.shape[-1])

def recall(output, target):
    pred = output.argmax(1)
    reca = 0
    for i in range(output.shape[-1]):
        p = pred==i
        g = target==i
        reca += (p*g).sum()/(g.sum()+1e-7)
    return reca.mul_(100.0 / output.shape[-1])

def f1_score(prec, reca):
    f1 = 2*(prec*reca)/(prec+reca)
    return f1

