import matplotlib.pyplot as plt
import numpy as np

def update_summary(summary, mode, iter, losses, errors, im, pc, gt, pred):

    for k in list(losses.keys()):
        summary.add_scalar(mode + '_loss/' + k, losses[k].avg, iter) 

    for k in list(errors.keys()):
        summary.add_scalar(mode + '_error/' + k, errors[k].avg, iter)   

    summary_img = image_draw(im, pc, gt, pred)  
    for k in list(summary_img.keys()):
        summary.add_image(mode + '_image/' + k, summary_img[k], iter)    

    return 

def image_draw(im, pc, gt, pred):   

    img_rgb = im.cpu().detach().numpy()[0]
    img_gt = pc_to_depth_img_numpy(gt)
    img_pc = pc_to_depth_img_numpy(pc)
    img_pred = pc_to_depth_img_numpy(pred['depth'])

    img_im_mask = pc_to_depth_img_numpy(pred['im_mask'])
    img_pc_mask = pc_to_depth_img_numpy(pred['pc_mask'])    

    summary_img = {
        'rgb': img_rgb.astype('uint8'),
        'pc': img_pc.astype('uint8'),
        'gt': img_gt.astype('uint8'),
        'pred': img_pred.astype('uint8'),

        'im_mask': img_im_mask.astype('uint8'),
        'pc_mask': img_pc_mask.astype('uint8'),
        
    }
    return summary_img

def pc_to_depth_img_numpy(im):
    '''
    :param im: (B, 1, H, W)
    :return:
    '''
    if type(im) != np.ndarray:
        im = im.cpu().detach().numpy()
    im1 = im[0, 0, :, :]
    im1 = (im1 - np.min(im1)) / (np.max(im1) - np.min(im1))
    im1 = 255 * plt.cm.jet(im1)[:, :, :3]  # H, W, C
    im1 = np.transpose(im1, (2, 0, 1))
    return im1.astype('uint8')