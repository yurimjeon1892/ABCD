import os
import random
import numpy as np
import torch.utils.data as data

from PIL import Image
from collections import namedtuple

from .generate_data import GenerateData

__all__ = ['Virtual2']

class Virtual2(data.Dataset):
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
        self.process_data = ProcessVirtual2(args)
        self.generate_data = GenerateData(args)
        self.data_path = args['data_root']

        # self.weathers = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
        self.weather = args['weather']
        self.dates = ['0001', '0002', '0006', '0018', '0020']
        self.num_samples = args['val_samples']
        self.samples = self.make_dataset()
        
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_path + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pcd, calib, rgb_img, d_img = \
            self.file_reader(self.samples[index])
        # print('file_reader:', pcd.shape, rgb_img.shape, d_img.shape)
        pc1, im2, pc3, gt = \
            self.process_data(pcd, calib, rgb_img, d_img, self.samples[index]['camnum'])
        # print('process_data:', pc1.shape, im2.shape, pc3.shape, gt.shape)
        pc1, generated_data = self.generate_data(pc1)
        return pc1, im2, pc3, gt, generated_data, self.samples[index]['fname']
    
    def make_dataset(self):
        valid_sample_list = []
        for date in self.dates:
            file_list = os.listdir(os.path.join(self.data_path, 'KITTI', 'training', 'velodyne', date))
            scenenum = 'Scene' + date[-2:]
            for fname in file_list: # 000000.bin
                frame = fname[:6]
                for cam in ['Camera_0', 'Camera_1']:
                    weather = self.weather
                    valid_sample = {'calib': os.path.join(self.data_path, 'KITTI', 'training', 'calib', date + '.txt'),
                                    'velodyne_raw': os.path.join(self.data_path, 'KITTI', 'training', 'velodyne', date,
                                                                 frame + '.bin'),
                                    'image': os.path.join(self.data_path, 'vkitti2', scenenum, weather, 'frames', 'rgb',
                                                          cam, 'rgb_' + frame[1:] + '.jpg'),
                                    'groundtruth': os.path.join(self.data_path, 'vkitti2', scenenum, weather, 'frames',
                                                                'depth', cam, 'depth_' + frame[1:] + '.png'),
                                    'camnum': cam,
                                    'fname': scenenum + '_' + weather + '_' + cam + '_' + frame + '.png'}
                    valid_sample_list.append(valid_sample)
        random.shuffle(valid_sample_list)
        if self.num_samples > 0:
            valid_sample_list = valid_sample_list[:self.num_samples]
        return valid_sample_list
    
    def rgb_read(self, filename):
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
        return rgb_png

    def depth_read(self, filename):
        # loads depth map D from png file
        # and returns it as a numpy array,
        # for details see readme.txt
        assert os.path.exists(filename), "file not found: {}".format(filename)
        img_file = Image.open(filename)
        depth_png = np.array(img_file)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)
        depth = depth_png.astype('uint16')
        return depth

    def pcd_read(self, filename):
        scan = np.fromfile(filename, dtype=np.float32)
        return scan.reshape((-1, 4))

    def calib_read(self, filename):
        data = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                line_value = line.split(' ')
                if line_value[0] in ['P0:', 'P1:', 'P2:', 'P3:']:
                    data[line_value[0][:2]] = np.array([float(x) for x in line_value[1:-2]])
                elif line_value[0] in ['R_rect', 'Tr_velo_cam', 'Tr_imu_velo']:
                    data[line_value[0]] = np.array([float(x) for x in line_value[1:-2]])

        data['P0'] = np.reshape(data['P0'], (3, 4))
        data['P1'] = np.reshape(data['P1'], (3, 4))
        data['P2'] = np.reshape(data['P2'], (3, 4))
        data['P3'] = np.reshape(data['P3'], (3, 4))

        data['R0_rect'] = np.eye(4)
        data['R0_rect'][:3, :3] = np.reshape(data['R_rect'], (3, 3))

        data['Tr_velo_to_cam'] = np.reshape(data['Tr_velo_cam'], (3, 4))
        data['Tr_velo_to_cam'] = np.vstack([data['Tr_velo_to_cam'], [0, 0, 0, 1]])

        calib = namedtuple('CalibData', data.keys())(*data.values())
        return calib

    def file_reader(self, sample_path):
        """
        :param sample_path:
        :return:
        """

        pcd = self.pcd_read(sample_path['velodyne_raw'])
        rgb_img = self.rgb_read(sample_path['image'])
        calib = self.calib_read(sample_path['calib'])
        d_img = self.depth_read(sample_path['groundtruth'])

        return pcd, calib, rgb_img, d_img

import math

class ProcessVirtual2(object):

    def __init__(self, args):
        self.num_points = args['num_points']
        return

    def __call__(self, pcd, calib, rgb_img, d_img, camnum):

        d_img[d_img == np.max(d_img)] = 0

        s_mask, h_min = self.velo2mask(pcd, calib, rgb_img, camnum)
        s_img = np.zeros(shape=d_img.shape, dtype=np.float32)
        s_img[s_mask] = d_img[s_mask]

        d_img[:h_min, :] = 0

        s_img = self.crop_im(s_img)
        rgb_img = self.crop_im(rgb_img)
        d_img = self.crop_im(d_img)

        sparse_pnts = self.im2pc(s_img)
        if self.num_points < sparse_pnts.shape[0]:
            sampled_indices1 = \
                np.random.choice(range(sparse_pnts.shape[0]),
                                 size=self.num_points, replace=False, p=None)
            pc1 = sparse_pnts[sampled_indices1].T
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

    def velo2mask(self, pcd, calib, rgb_img, camnum):
        """
            Color pointcloud by assigning RGB pixel values
            at projected points on the image
            :param pcd: NumPy array of xyz in shape (N, 3)
            :param calib: Calibration information
            :param rgb_img: Pillow Image instance
            in range [0, 1] if true, [0, 255] otherwise
            :return xyz: NumPy array of xyz in shape (M, 3)
            :return rgb: NumPy array of rgb in shape (M, 3)
        """
        # Replace intensity to 1s
        pcd = pcd.T
        num_pts = pcd.shape[1]
        pcd[-1, :] = np.ones((1, num_pts))

        # Transform to cam frame
        # d_img = P2 * R0_rect * Tr_velo_to_cam * velodyne_raw
        # pts_2d = np.dot(calib.T_cam2_velo, pcd)
        if camnum == 'Camera_0':
            pts_2d = np.dot(calib.P2, calib.R0_rect)
        elif camnum == 'Camera_1':
            pts_2d = np.dot(calib.P3, calib.R0_rect)

        pts_2d = np.dot(pts_2d, calib.Tr_velo_to_cam)
        pts_2d = np.dot(pts_2d, pcd)

        height, width = rgb_img.shape[0], rgb_img.shape[1]
        s_mask = np.zeros(shape=(height, width), dtype=np.bool)

        h_min = height
        pts_2d = pts_2d.T
        for idx, xyw in enumerate(pts_2d):
            x = xyw[0]
            y = xyw[1]
            w = xyw[2]
            is_in_img = (
                    w > 0 and 0 <= x < w * width and 0 <= y < w * height
            )

            if is_in_img:
                s_mask[int(y / w), int(x / w)] = True
                if h_min > int(y / w): h_min = int(y / w)

        return s_mask, h_min

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

    def crop_im(self, img):
        oheight, owidth = 352, 1216
        h = img.shape[0]
        w = img.shape[1]
        th, tw = oheight, owidth
        i = h - th
        j = int(math.floor((w - tw) / 2.))

        if img.ndim == 3:
            return img[i:i + th, j:j + tw, :]
        elif img.ndim == 2:
            return img[i:i + th, j:j + tw]

    # def crop_im(self, img, th=352, tw=1216):
    #     i, j = 0, 0
    #     if img.ndim == 3:
    #         return img[i:i + th, j:j + tw, :]
    #     elif img.ndim == 2:
    #         return img[i:i + th, j:j + tw]