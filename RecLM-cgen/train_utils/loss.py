import torch
import torch.nn as nn


class CrossEntropyLoss_e(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, ignore_value=-100, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ignore_value = ignore_value

    def forward_3(self, logits, labels):
        C = logits.shape[1]

        mask = (labels != self.ignore_value).float()

        labels = torch.where(labels == self.ignore_value, torch.zeros_like(labels), labels)

        offset_logits = logits - logits.max(dim=1, keepdim=True).values

        exp_logits = offset_logits.exp()

        s = exp_logits.sum(dim=1, keepdim=True)

        log_prob = offset_logits - s.log()

        if self.weight != None:
            weight = self.weight.view(1, C, 1).repeat(logits.shape[0], 1, logits.shape[2])
            log_prob *= weight

            if self.ignore_index in range(0, C):
                weight[:, self.ignore_index] = 0

            w_y = torch.gather(weight, dim=1, index=labels.unsqueeze(1))
            weightSum = w_y.view(-1).sum()

        if self.ignore_index in range(0, C):
            log_prob[:, self.ignore_index] = 0

        l = - torch.gather(log_prob, dim=1, index=labels.unsqueeze(1)).squeeze()
        l = l * mask
        if self.weight == None:
            loss = l.sum() / mask.sum()
        else:
            weightSum = (l * mask).sum() / mask.sum()
            loss = weightSum
        return loss

    def forward_1(self, logits, labels, scope_mask):
        C = logits.shape[1]

        mask = (labels != self.ignore_value).float()

        labels = torch.where(labels == self.ignore_value, torch.zeros_like(labels), labels)

        offset_logits = logits - logits.max(dim=1, keepdim=True).values
        exp_logits = offset_logits.exp()

        exp_logits_label = torch.gather(exp_logits, dim=1, index=labels.unsqueeze(1))

        s = exp_logits.sum(dim=1, keepdim=True)

        probability_label = torch.div(exp_logits_label, s)
        gamma = self.gamma
        shrink_coe = torch.pow((1-probability_label), gamma)
        scope_positions = scope_mask.any(dim=1)
        shrink_coe[~scope_positions] = 1

        log_prob = offset_logits - s.log()

        if self.weight != None:
            weight = self.weight.view(1, C, 1).repeat(logits.shape[0], 1, logits.shape[2])
            log_prob *= weight

            if self.ignore_index in range(0, C):
                weight[:, self.ignore_index] = 0

            w_y = torch.gather(weight, dim=1, index=labels.unsqueeze(1))
            weightSum = w_y.view(-1).sum()

        if self.ignore_index in range(0, C):
            log_prob[:, self.ignore_index] = 0

        l = - torch.gather(log_prob, dim=1, index=labels.unsqueeze(1)).squeeze()
        l = l.mul(shrink_coe.view(-1))
        l = l * mask
        if self.weight == None:
            loss = l.sum() / mask.sum()
        else:
            weightSum = (l * mask).sum() / mask.sum()
            loss = weightSum
        return loss

    def forward_2(self, logits, labels, scope_mask):
        C = logits.shape[1]

        mask = (labels != self.ignore_value).float()

        labels = torch.where(labels == self.ignore_value, torch.zeros_like(labels), labels)


        offset_logits = logits - logits.max(dim=1, keepdim=True).values
        exp_logits = offset_logits.exp()

        exp_logits_label = torch.gather(exp_logits, dim=1, index=labels.unsqueeze(1))

        s = exp_logits.sum(dim=1, keepdim=True)

        probability_label = torch.div(exp_logits_label, s)
        gamma = self.gamma
        shrink_coe = torch.pow((1-probability_label), gamma)

        log_prob = offset_logits - s.log()

        if self.weight != None:
            weight = self.weight.view(1, C, 1).repeat(logits.shape[0], 1, logits.shape[2])
            log_prob *= weight

            if self.ignore_index in range(0, C):
                weight[:, self.ignore_index] = 0

            w_y = torch.gather(weight, dim=1, index=labels.unsqueeze(1))
            weightSum = w_y.view(-1).sum()

        if self.ignore_index in range(0, C):
            log_prob[:, self.ignore_index] = 0

        l = - torch.gather(log_prob, dim=1, index=labels.unsqueeze(1)).squeeze()
        l = l.mul(shrink_coe.view(-1))
        l = l * mask
        if self.weight == None:
            loss = l.sum() / mask.sum()
        else:
            weightSum = (l * mask).sum() / mask.sum()
            loss = weightSum
        return loss