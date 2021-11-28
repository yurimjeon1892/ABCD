import os, sys, gc

import torch.utils.data
from tqdm import tqdm
import numpy as np
import cv2

from utils.helper import Err

import datetime

def test(dloader, model, args):
    
    ckpt_dir = args['resume'].split('/')[-2] + '_' + str(args['iter']) + '_test'
    save_dir = os.path.join('../results', ckpt_dir)
    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    model.eval()
    with torch.no_grad():
        description = '[i] Test'
        for i, (pc, im, d, gt, generated_data, fname) in \
                enumerate(tqdm(dloader, desc=description, unit='batches')):
            # Convert data type
            pc = pc.to(args['DEVICE']).float()
            im = im.to(args['DEVICE']).float()
            d = d.to(args['DEVICE']).float()

            # run model
            pred = model(pc, im, d, generated_data,
                        (list(d.size())[2], list(d.size())[3]))

            # uint16 depth image
            img_pred = pred['depth'][0].cpu().detach().numpy() * (2 ** 8)
            img_pred = np.squeeze(np.clip(img_pred, 0, 2 ** 16 - 1))
            img_pred = img_pred.astype('uint16')
            cv2.imwrite(os.path.join(save_dir, fname[0]), img_pred)

            # del pc1, im2, pc3, gt, generated_data, pred
            # torch.cuda.empty_cache()

    print("[i] Test ended. ")
    return

def val(dloader, model, args):

    save_dir = None

    if args['save_image']:
        save_dir = datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M")
        save_dir = os.path.join('../', save_dir)
        os.makedirs(save_dir, mode=0o777, exist_ok=True)        

    err = Err(args['dataset'])

    model.eval()
    with torch.no_grad():
        description = '[i] Validate'
        for i, (pc, im, d, gt, generated_data, fname) in \
                enumerate(tqdm(dloader, desc=description, unit='batches')):
            # Convert data type
            try:
                pc = pc.to(args['DEVICE']).float()
                im = im.to(args['DEVICE']).float()
                d = d.to(args['DEVICE']).float()
                gt = gt.to(args['DEVICE']).float()

                # run model
                pred = model(pc, im, d, generated_data,
                            (list(gt.size())[2], list(gt.size())[3]))

                err.update(gt, pred, pc.size(0))

                if args['save_image']:           

                    img_pred = pred['depth'][0].cpu().detach().numpy() * (2 ** 8)
                    img_pred = np.squeeze(np.clip(img_pred, 0, 2 ** 16 - 1))
                    img_pred = img_pred.astype('uint16')
                    cv2.imwrite(os.path.join(save_dir, fname[0]), img_pred)

                del pc, im, d, gt, generated_data, pred
                torch.cuda.empty_cache()

            except RuntimeError as ex:
                if "CUDA out of memory" in str(ex) or "cuda runtime error" in str(ex):
                    print("out of memory, continue")
                    del pc, im, d, gt, generated_data
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    sys.exit(1)

    print("[i] Validate ended. ")

    print('Error; ', end=" ")
    for k in list(err.dict.keys()):
        print(k + ' {:.4f}'.format(err.dict[k].avg), end=" ")
    print()
    return