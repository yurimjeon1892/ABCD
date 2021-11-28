import sys
import gc
import traceback

import torch.nn.parallel
import torch.optim
import torch.utils.data

from tqdm import tqdm
from utils.helper import Lss, Err
from utils.summary import update_summary


def validate(val_loader, model, criterion, args, summary, it):

    lss = Lss(criterion.loss_name)
    err = Err(args['dataset'])

    model.eval()
    with torch.no_grad():
        description = '[i] Valid iter {}'.format(it)
        for i, (pc, im, d, gt, generated_data, fname) in \
                enumerate(tqdm(val_loader, desc=description, unit='batches')):
            try:
                # Convert data type
                pc = pc.to(args['DEVICE']).float()
                im = im.to(args['DEVICE']).float()
                d = d.to(args['DEVICE']).float()
                gt = gt.to(args['DEVICE']).float()

                # run model
                pred = model(pc, im, d, generated_data,
                            (list(gt.size())[2], list(gt.size())[3]))
                
                # compute loss
                losses = criterion.compute_loss(gt, pred)

                lss.update(losses, pc.size(0))
                err.update(gt, pred, pc.size(0))

            except RuntimeError as ex:
                print("in VAL, RuntimeError " + repr(ex))
                # traceback.print_tb(ex.__traceback__, file=logger.out_fd)
                traceback.print_tb(ex.__traceback__)

                if "CUDA out of memory" in str(ex) or "cuda runtime error" in str(ex):
                    print("out of memory, continue")
                    del pc, im, d, gt, generated_data, pred
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    sys.exit(1)

    update_summary(summary, 'valid', it, lss.dict, err.dict, im, d, gt, pred)

    print('[i] Valid iter {}; '.format(it))
    print('Loss; ', end=" ")
    for k in list(lss.keys):
        print(k + ' {:.2f}'.format(lss.dict[k].avg), end=" ")
    print()
    print('Error; ', end=" ")
    for k in list(err.dict.keys()):
        print(k + ' {:.4f}'.format(err.dict[k].avg), end=" ")
    print()

    return lss.dict