from torch import nn
import math
import torch
import numpy as np
import torch.nn.functional as F

class MeanResidueLossAdaptive(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_age, end_age, K=6):
        super().__init__()
        np.random.seed(2019)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age
        self.K = K

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)

        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # resdue loss ---- K is not constant
        EPS = 1e-3
        width = self.end_age + 1 - self.start_age
        pos = torch.zeros(N, width).cuda()
        # print(N, width)
        # if N == 1:
        #     import pdb; pdb.set_trace()

        for i in range(N):
            try:
                pos[i, target[i].int()] = 1
            except:
                pos[i, int(target.item())] = 1

        prob_gt = (pos * p).sum(1).unsqueeze(dim=1).cuda() # probs of the gt. dim 1 is squeezed

        prob_gt = prob_gt.expand(N,width) #.t() why transpose ?
        pos_no_K = torch.tensor(p < prob_gt, dtype = float).cuda()# int)

        p_not_K = pos_no_K * p
        residue_loss = (-(p_not_K + EPS) * torch.log(p_not_K + EPS)).sum(1,keepdim = True).mean()
        # compute k
        batch_average_K = float(((pos_no_K == 0).sum()/N).item())

        return self.lambda_1 * mean_loss, self.lambda_2 * residue_loss, batch_average_K