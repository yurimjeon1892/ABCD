import os
import random
import numpy as np
import torch.utils.data as data

from PIL import Image

from .generate_data import GenerateData

__all__ = ['KITTI']


class KITTI(data.Dataset):
    """
    Args:
        split:
        process_data (callable):
        generate_data (callable):
        args:
    """

    def __init__(self,
                 split,
                 args):
        self.split = split
        self.process_data = ProcessKITTI(split, args)
        self.generate_data = GenerateData(args)
        self.data_root = args['data_root']
        if split == 'train':
            self.dates = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
            self.data_path = args['data_root']
            self.num_samples = args['train_samples']
            self.samples = self.make_train_dataset()
        elif split == 'valid':
            self.data_path = os.path.join(args['data_root'], 'depth_selection',
                                          'val_selection_cropped')
            self.num_samples = args['val_samples']
            self.samples = self.make_valid_dataset()
        elif split == 'test':
            self.data_path = os.path.join(args['data_root'], 'depth_selection',
                                          'test_depth_completion_anonymous')
            self.num_samples = 1000
            self.samples = self.make_test_dataset()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s_img, rgb_img, d_img, fname = \
            self.file_reader(self.samples[index])
        pc1, im2, pc3, gt = \
            self.process_data(s_img, rgb_img, d_img)
        pc1, generated_data = self.generate_data(pc1)
        return pc1, im2, pc3, gt, generated_data, fname

    def make_train_dataset(self):  
        train_sample_list = []
        for date in self.dates:            
            folder_list = os.listdir(os.path.join(self.data_path, date))     
            for folder in folder_list:
                if folder[-4:] == '.txt':
                    continue
                if os.path.isdir(os.path.join(self.data_path, 'train', folder)):
                    split = 'train'
                elif os.path.isdir(os.path.join(self.data_path, 'val', folder)):
                    split = 'val'
                else:
                    continue
                file2_list = os.listdir(os.path.join(self.data_path, split, folder, 'proj_depth', 'groundtruth', 'image_02'))
                for fname in file2_list:                        
                    train_sample = {'image': os.path.join(self.data_path, date, folder, 'image_02', 'data', fname),
                                    'velodyne_raw': os.path.join(self.data_path, split, folder, 'proj_depth', 'velodyne_raw', 'image_02', fname),
                                    'groundtruth': os.path.join(self.data_path, split, folder, 'proj_depth', 'groundtruth', 'image_02', fname),
                                    'fname': fname}
                    train_sample_list.append(train_sample)

                file3_list = os.listdir(os.path.join(self.data_path, split, folder, 'proj_depth', 'groundtruth', 'image_03'))
                for fname in file3_list:                        
                    train_sample = {'image': os.path.join(self.data_path, date, folder, 'image_03', 'data', fname),
                                    'velodyne_raw': os.path.join(self.data_path, split, folder, 'proj_depth', 'velodyne_raw', 'image_03', fname),
                                    'groundtruth': os.path.join(self.data_path, split, folder, 'proj_depth', 'groundtruth', 'image_03', fname),
                                    'fname': fname}
                    train_sample_list.append(train_sample)    
        random.shuffle(train_sample_list)
        if self.num_samples > 0:
            train_sample_list = train_sample_list[:self.num_samples]
        else:
            self.num_samples = len(train_sample_list)
        return train_sample_list
    
    def make_valid_dataset(self):        
        # 2011_10_03_drive_0047_sync_velodyne_raw_0000000743_image_02.png
        # 2011_10_03_drive_0047_sync_groundtruth_depth_0000000764_image_03.png
        # 2011_10_03_drive_0047_sync_image_0000000710_image_03.png
        valid_sample_list = []
        file_list = os.listdir(os.path.join(self.data_path, 'image'))          
        for fname in file_list:
            prefix = fname[:26]
            postfix = fname[-23:]
            valid_sample = {'image': os.path.join(self.data_path, 'image', prefix + '_image_' + postfix),
                            'velodyne_raw': os.path.join(self.data_path, 'velodyne_raw', prefix + '_velodyne_raw_' + postfix),
                            'groundtruth': os.path.join(self.data_path, 'groundtruth_depth', prefix + '_groundtruth_depth_' + postfix),
                            'fname': fname
                            }
            valid_sample_list.append(valid_sample)

        random.shuffle(valid_sample_list)
        if self.num_samples > 0:
            valid_sample_list = valid_sample_list[:self.num_samples]
        return valid_sample_list
    
    def make_test_dataset(self):  
        test_sample_list = []
        file_list = os.listdir(os.path.join(self.data_path, 'image'))        
        for fname in file_list:
            test_sample = {'image': os.path.join(self.data_path, 'image', fname),
                           'velodyne_raw': os.path.join(self.data_path, 'velodyne_raw', fname),
                           'fname': fname
                          }
            test_sample_list.append(test_sample)
        return test_sample_list
    
    def rgb_read(self, filename):
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
        rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
#         img_file.close()
        return rgb_png

    def depth_read(self, filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        depth_png = np.array(img_file)
#         img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)
        
        depth = depth_png.astype('uint16')
        return depth

    def file_reader(self, sample_path):
        """
        Args:
            path:
        Returns:
            s_img: ndarray (H, W) np.float32
            rgb_img: ndarray (H, W, 3) np.float32
            d_img: ndarray (H, W) np.float32
        """

        s_img = self.depth_read(sample_path['velodyne_raw'])
        rgb_img = self.rgb_read(sample_path['image'])
        if 'groundtruth' in sample_path:
            d_img = self.depth_read(sample_path['groundtruth'])
        else: d_img = s_img
        return s_img, rgb_img, d_img, sample_path['fname']

import math

oheight, owidth = 352, 1216

class ProcessKITTI(object):

    def __init__(self, split, args):
        self.split = split
        self.num_points = args['num_points']
        return

    def __call__(self, s_img, rgb_img, d_img):

        s_img = self.crop_im(s_img)
        rgb_img = self.crop_im(rgb_img)
        d_img = self.crop_im(d_img)

        sparse_pnts = self.im2pc(s_img)
        if self.num_points < sparse_pnts.shape[0]:
            # sampled_indices1 = \
            #     np.random.choice(range(sparse_pnts.shape[0]),
            #                      size=self.num_points, replace=False, p=None)
            # pc1 = sparse_pnts[sampled_indices1].T
            pc1 = sparse_pnts[:self.num_points, :].T
        else:
            pc1 = np.zeros(shape=(3, self.num_points))
            pc1[:, :sparse_pnts.shape[0]] = sparse_pnts.T

        im2 = np.transpose(rgb_img, (2, 0, 1))
        pc3 = np.expand_dims(s_img, 0) / (2 ** 8)
        gt = np.expand_dims(d_img, 0) / (2 ** 8)

        im2 = np.ascontiguousarray(im2, dtype=np.float32)
        pc3 = np.ascontiguousarray(pc3, dtype=np.float32)
        gt = np.ascontiguousarray(gt, dtype=np.float32)

        return pc1, im2, pc3, gt

    def im2pc(self, img):
        sparse_pnts = []
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                if img[h, w] > 0:
                    sparse_pnts.append([float(h),
                                        float(w),
                                        float(img[h, w])])
        sparse_pnts = np.array(sparse_pnts) / (2 ** 8)
        return sparse_pnts

    def crop_im(self, img, th=352, tw=1216):
        i, j = 0, 0
        if img.ndim == 3:
            return img[i:i + th, j:j + tw, :]
        elif img.ndim == 2:
            return img[i:i + th, j:j + tw]
