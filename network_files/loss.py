import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=1, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.size_average = size_average
#         self.elipson = 0.000001
#
#     def forward(self, logits, labels):
#         """
#         cal culates loss
#         logits: batch_size * labels_length * seq_length
#         labels: batch_size * seq_length
#         """
#         if labels.dim() > 2:
#             labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
#             labels = labels.transpose(1, 2)
#             labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
#         if logits.dim() > 3:
#             logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
#             logits = logits.transpose(2, 3)
#             logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
#         assert (logits.size(0) == labels.size(0))
#         assert (logits.size(2) == labels.size(1))
#         batch_size = logits.size(0)
#         labels_length = logits.size(1)
#         seq_length = logits.size(2)
#
#         # transpose labels into labels onehot
#         new_label = labels.unsqueeze(1)
#         label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
#
#         # calculate log
#         log_p = F.log_softmax(logits)
#         pt = label_onehot * log_p
#         sub_pt = 1 - pt
#         fl = -self.alpha * (sub_pt) ** self.gamma * log_p
#         if self.size_average:
#             return fl.mean()
#         else:
#             return fl.sum()


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss



# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#         The losses are averaged across observations for each minibatch.
#
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#
#
#     """
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         # print(class_mask)
#
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P*class_mask).sum(1).view(-1,1)
#
#         log_p = probs.log()
#         # print('probs size= {}'.format(probs.size()))
#          #print(probs)
#
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
#         # print('-----bacth_loss------')
#         # print(batch_loss)
#
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss

