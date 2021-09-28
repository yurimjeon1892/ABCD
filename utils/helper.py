# helper functions for training
import os, sys
import shutil
import numpy as np
import torch
import yaml

def load_config(resume, args):
    pathes = args['resume'].split('/')
    resumed_ckpt_path = ''
    for path_ in pathes[:-1]:
        resumed_ckpt_path = os.path.join(resumed_ckpt_path, path_)
    with open(os.path.join(resumed_ckpt_path, 'config.yaml'), 'r') as stream:
        resumed_args = yaml.safe_load(stream)
    args['arch'] = resumed_args['arch']
    args['last_relu'] = resumed_args['last_relu']
    args['use_leaky'] = resumed_args['use_leaky']
    args['bcn_use_bias'] = resumed_args['bcn_use_bias']
    args['bcn_use_norm'] = resumed_args['bcn_use_norm']
    args['DEVICE'] = resumed_args['DEVICE']
    args['dim'] = resumed_args['dim']
    args['batch_size'] = resumed_args['batch_size']
    args['scales_filter_map'] = resumed_args['scales_filter_map']
    args['pc_feat_channel'] = resumed_args['pc_feat_channel']
    args['resnet'] = resumed_args['resnet']
    return args

def adjust_learning_rate(lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.7 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        print('old lr', param_group['lr'])
        if lr < param_group['lr']:
            param_group['lr'] = lr
            print('new lr', param_group['lr'])
        else:
            lr = param_group['lr']
    return lr

# ---------------- Pretty sure the following functions/classes are common ----------------
def save_checkpoint(state, is_best, ckpt_dir,
                    filename='checkpoint.pth.tar',
                    iter_iterval=1000):
    torch.save(state, os.path.join(ckpt_dir, filename))
    if state['iter'] % iter_iterval == 0:
        shutil.copyfile(
            os.path.join(ckpt_dir, filename),
            os.path.join(ckpt_dir, 'checkpoint_' + str(state['iter'])+'.pth.tar'))
        print(state['iter'], 'iter checkpoint saved')

    if is_best:
        shutil.copyfile(
            os.path.join(ckpt_dir, filename),
            os.path.join(ckpt_dir, 'model_best.pth.tar'))

    if state['iter'] > 5 * iter_iterval:
        prev_checkpoint_filename = os.path.join(
            ckpt_dir, 'checkpoint_' + str(state['iter'] - 5 * iter_iterval) + '.pth.tar')
        # print('prev_checkpoint_filename', prev_checkpoint_filename)
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

class Lss():
    def __init__(self, keys):
        self.keys = keys
        self.dict = {}
        self.flush()  

    def flush(self):
        for k in self.keys:
            self.dict[k] = AverageMeter()      

    def update(self, losses, batch):
        for k in list(losses.keys()):
            self.dict[k].update(losses[k].item(), batch) 
        return 

class Err():
    def __init__(self, dataset):
        self.dict = {}
        self.errors = {}
        self.dataset = dataset
        self.flush()        

    def flush(self):
        for k in list(self.errors.keys()):
            self.dict[k] = AverageMeter()

    def metrics(self, gt, pred):

        rmse_ = self.rmse(gt, pred)
        mae_ = self.mae(gt, pred)
        irmse_ = self.irmse(gt, pred)
        imae_ = self.imae(gt, pred)

        errors = {
            'rmse': rmse_,
            'mae': mae_,
            'irmse': irmse_,
            'imae': imae_,
        } 

        if self.dataset != 'Virtual2':
            errors = {
            'rmse': rmse_ * 1000,
            'mae': mae_ * 1000,
            'irmse': irmse_ * 1000,
            'imae': imae_ * 1000,
            }
        self.errors.update(errors)

    def update(self, gt, pred, batch, ablation=False):

        gt_np = gt.cpu().detach().numpy()[0]
        pred_np = pred['depth'].cpu().detach().numpy()[0]

        if self.dataset == 'Virtual2': 
            self.metrics(gt_np * 256. / 100., pred_np * 256. / 100.)
        else: self.metrics(gt_np, pred_np)
            
        if not self.dict: self.flush()
    
        for k in list(self.errors.keys()):
            self.dict[k].update(self.errors[k], batch)         
        return    

    def rmse(self, gt, pred):
        dif = gt[np.where(gt>0.0)] - pred[np.where(gt>0.0)]
        error = np.sqrt(np.mean(dif**2))
        return error

    def mae(self, gt, pred):
        dif = gt[np.where(gt>0.0)] - pred[np.where(gt>0.0)]
        error = np.mean(np.fabs(dif))
        return error

    def irmse(self, gt, pred):
        dif = 1.0/gt[np.where(gt>0.0)] - 1.0/pred[np.where(gt>0.0)]
        error = np.sqrt(np.mean(dif**2))
        return error

    def imae(self, gt, pred):
        dif = 1.0/gt[np.where(gt>0.0)] - 1.0/pred[np.where(gt>0.0)]
        error = np.mean(np.fabs(dif))
        return error
