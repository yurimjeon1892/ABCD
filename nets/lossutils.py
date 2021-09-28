import torch.nn as nn

class MSE(nn.Module):
    def __init__(self, args):
        super(MSE, self).__init__()

        self.loss_name = ['mse']         
    
    def compute(self, gt, pred):
        '''
        :param gt: (B, 1, H, W)
        :param pred: (B, 1, H, W)
        :return:
        '''
        assert pred.dim() == gt.dim(), "inconsistent dimensions"
        valid_mask = (gt > 0).detach()
        diff = gt - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()

        losses = {
            'mse': loss
        }

        return losses