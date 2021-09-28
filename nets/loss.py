from .lossutils import *

class Criterion(nn.Module):

    def __init__(self, args):
        super(Criterion, self).__init__()
        
        self.mse = MSE(args)       

        self.loss_name = ['total']
        self.loss_name += self.mse.loss_name

        print('=> complute loss list: ', self.loss_name)

    def compute_loss(self, gt, pred):

        losses = {}
        mseloss = self.mse.compute(gt, pred['depth'])
        losses.update(mseloss)

        total = 0
        for k in losses.keys():
            total += losses[k]
        losses['total'] = total

        return losses