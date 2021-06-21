import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, smoothing=0.0, multi_label=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.smoothing = smoothing
        self.confidence = 1-smoothing
        self.multi_label = multi_label

        # self.multi_label_mapping = {1: [2], 3: [4], 5: [6]}
        #
        # for key, ind_list in self.multi_label_mapping.items():
        #     src_matrix = torch.zeros(class_num)
        #     avg_value = 1 / (1 + len(ind_list))
        #     every_ind = torch.tensor(ind_list + [key])
        #     src_matrix.scatter_(0, every_ind, avg_value)
        #     self.multi_label_mapping[key] = (ind_list, src_matrix)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(self.smoothing / (self.class_num - 1))
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, self.confidence)

        # for i in range(N):
        #     label = int(ids[i][0])
        #     if label not in self.multi_label_mapping:
        #         class_mask[i] = torch.zeros(self.class_num).scatter_(0, ids[i][0], 1)
        #     else:
        #         class_mask[i] = self.multi_label_mapping[label][1]
        #
        # print(ids)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__ == '__main__':
    output = Variable(torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float))
    target = Variable(torch.tensor([[0], [1]], dtype=torch.long))
    loss = FocalLoss(8)
    print(loss.forward(output, target))
