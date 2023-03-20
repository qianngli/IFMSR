import torch.nn.functional as F
from torch import nn
import pdb

class HEL(nn.Module):
    def __init__(self):
        super(HEL, self).__init__()
        print("You are using `HEL`!")
        self.eps = 1e-6
        self.image = nn.L1Loss() 

    def edge_loss(self, pred, target):

        edge = target - F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        edge[edge != 0] = 1
        # input, kernel_size, stride=None, padding=0
        numerator = (edge * (pred - target).abs_()).sum([2, 3])
        denominator = edge.sum([2, 3]) + self.eps
        return numerator / denominator

    def forward(self, pred, target):
        edge_loss = self.edge_loss(pred, target).mean()
        sr_loss = self.image(pred, target)
        pdb.set_trace()
        return 0.1*edge_loss + sr_loss