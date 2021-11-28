import sys
import gc
import traceback

import torch.nn.parallel
import torch.optim
import torch.utils.data

from tqdm import tqdm
from valid import validate

from utils.helper import adjust_learning_rate, save_checkpoint, Lss, Err
from utils.summary import update_summary

def iterater(loader, model, criterion, optimizer, args, summary, it_dict):

    do_eval = True
    curr_epoch = (it_dict['iter'] // len(loader['train'])) + 1

    lss = Lss(criterion.loss_name)
    err = Err(args['dataset'])    

    adjust_learning_rate(args['lr'], optimizer, curr_epoch)
    model.train()
    description = '[i] Train {:>2}/{}'.format(curr_epoch, args['epochs'])
    for i, (pc, im, d, gt, generated_data, fname) in \
            enumerate(tqdm(loader['train'], desc=description, unit='batches')):
        try:            
            # Convert data type
            pc = pc.to(args['DEVICE']).float()
            im = im.to(args['DEVICE']).float()
            d = d.to(args['DEVICE']).float()
            gt = gt.to(args['DEVICE']).float()

            # run model
            pred = model(pc, im, d, generated_data,
                         (list(gt.size())[2], list(gt.size())[3]),
                         it_dict['iter'] == 0)
            
            # calculate loss
            losses = criterion.compute_loss(gt, pred)

            # backprop
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            lss.update(losses, pc.size(0))
            err.update(gt, pred, pc.size(0))

            if it_dict['iter'] % args['iter_iterval'] == 0 and \
                    it_dict['iter'] != 0:

                update_summary(summary, 'train', it_dict['iter'], lss.dict, err.dict, im, d, gt, pred)

                print('[i] Train iter {}; '.format(it_dict['iter']))
                print('Loss; ', end=" ")
                for k in list(lss.keys):
                    print(k + ' {:.2f}'.format(lss.dict[k].avg), end=" ")
                print()
                print('Error; ', end=" ")
                for k in list(err.dict.keys()):
                    print(k + ' {:.4f}'.format(err.dict[k].avg), end=" ")
                print()

                is_train_best = True if it_dict['best_train_iter'] is None \
                    else (lss.dict['total'].avg < it_dict['min_train_loss'])
                if is_train_best:
                    it_dict['min_train_loss'] = lss.dict['total'].avg
                    it_dict['best_train_iter'] = it_dict['iter']

                if do_eval:
                    val_loss_dict = \
                        validate(loader['validate'], model, criterion, args, summary, it_dict['iter'])
                    gc.collect()

                    is_val_best = True if it_dict['best_val_iter'] is None \
                        else (val_loss_dict['total'].avg < it_dict['min_val_loss'])
                    if is_val_best:
                        it_dict['min_val_loss'] = val_loss_dict['total'].avg
                        it_dict['best_val_iter'] = it_dict['iter']
                        print("New min val loss!")

                min_loss = it_dict['min_val_loss'] if do_eval else it_dict['min_train_loss']
                is_best = is_val_best if do_eval else is_train_best
                save_checkpoint({
                    'iter': it_dict['iter'],  # next start epoch
                    'state_dict': model.state_dict(),
                    'min_loss': min_loss,
                    'optimizer': optimizer.state_dict(),
                    }, 
                    is_best,
                    args['ckpt_dir'],
                    iter_iterval=args['iter_iterval'])

                train_str = 'Best train loss: {:.5f} at iter {:3d}'.format(it_dict['min_train_loss'],
                                                                           it_dict['best_train_iter'])
                print(train_str)

                if do_eval:
                    val_str = 'Best val loss: {:.5f} at iter {:3d}'.format(it_dict['min_val_loss'],
                                                                           it_dict['best_val_iter'])
                    print(val_str)

                lss.flush()
                err.flush()

            it_dict['iter'] = it_dict['iter'] + 1

            #del pc, im, d, gt, generated_data, pred
            #torch.cuda.empty_cache()

        except RuntimeError as ex:
            print("in TRAIN, RuntimeError " + repr(ex))
            traceback.print_tb(ex.__traceback__)

            if "CUDA out of memory" in str(ex) or "cuda runtime error" in str(ex):
                print("out of memory, continue")
                del pc, im, d, gt, generated_data, pred
                torch.cuda.empty_cache()
                gc.collect()
            else:
                sys.exit(1)

    return it_dict