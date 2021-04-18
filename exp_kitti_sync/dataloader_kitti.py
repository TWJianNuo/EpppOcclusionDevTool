from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image, ImageFile
from kitti_utils import read_calib_file, labels, translateTrainIdSemantics

import torch.utils.data as data
from torchvision import transforms

from utils import *

import time

ImageFile.LOAD_TRUNCATED_IMAGES = True # This can prevent some weried error
class KittiDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 gt_path,
                 filenames,
                 height,
                 width,
                 crph=365,
                 crpw=1220,
                 is_train=False,
                 ):
        super(KittiDataset, self).__init__()

        self.data_path = data_path
        self.gt_path = gt_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()
        self.interp = Image.ANTIALIAS
        self.img_ext = '.png'

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.dirmapping = {'l': 'image_02', 'r': 'image_03'}

        self.crph = crph
        self.crpw = crpw

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)

        self.rescaleK = np.eye(4)
        self.rescaleK[0, 0] = self.width / self.crpw
        self.rescaleK[1, 1] = self.height / self.crph

    def preprocess(self, inputs, color_aug, rndseed):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # Do random Crop
        cropped_color = self.rndcrop_color(inputs["color"], rndseed)
        cropped_depth, intrinsic = self.rndcrop_depth(inputs["depthgt"], rndseed, inputs['K'])

        inputs["color"] = self.resize(cropped_color)
        inputs["depthgt"] = cropped_depth

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                inputs[k] = self.to_tensor(f)
                inputs[k + "_aug"] = self.to_tensor(color_aug(f))

            elif "depthgt" in k:
                inputs[k] = np.array(f).astype(np.float32) / 256.0
                inputs[k] = torch.from_numpy(inputs[k]).unsqueeze(0)
                inputs["K"] = torch.from_numpy(intrinsic.astype(np.float32))
                inputs["K_scaled"] = torch.from_numpy((self.rescaleK @ intrinsic).astype(np.float32))

    def __len__(self):
        return len(self.filenames)

    def rndcrop_color(self, img, rndseed):
        np.random.seed(rndseed)

        w, h = img.size

        if self.is_train:
            left = np.random.randint(w - self.crpw + 1)
            top = np.random.randint(h - self.crph + 1)
        else:
            left = int((w - self.crpw) / 2)
            top = int((h - self.crph) / 2)
        imgcropped = img.crop((left, top, left + self.crpw, top + self.crph))

        return imgcropped

    def rndcrop_depth(self, img, rndseed, intrinsic):
        np.random.seed(rndseed)

        w, h = img.size

        if self.is_train:
            left = np.random.randint(w - self.crpw + 1)
            top = np.random.randint(h - self.crph + 1)
        else:
            left = int((w - self.crpw) / 2)
            top = int((h - self.crph) / 2)
        imgcropped = img.crop((left, top, left + self.crpw, top + self.crph))

        intrinsic_cropped = np.copy(intrinsic)
        intrinsic_cropped[0, 2] = intrinsic_cropped[0, 2] - left
        intrinsic_cropped[1, 2] = intrinsic_cropped[1, 2] - top

        # Check
        # xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        # xx = xx[np.array(img) > 0]
        # yy = yy[np.array(img) > 0]
        #
        # xx_old = xx[200]
        # yy_old = yy[200]
        # d_old = float(np.array(img)[yy_old, xx_old]) / 256.0
        # pts_old = np.array([[xx_old * d_old, yy_old * d_old, d_old, 1]]).T
        # pts_old = np.linalg.inv(intrinsic) @ pts_old
        #
        # xx_new = xx_old - left
        # yy_new = yy_old - top
        # d_new = float(np.array(imgcropped)[yy_new, xx_new]) / 256.0
        # pts_new = np.array([[xx_new * d_new, yy_new * d_new, d_new, 1]]).T
        # pts_new = np.linalg.inv(intrinsic_cropped) @ pts_new
        #
        # assert np.abs(pts_new - pts_old).max() < 1e-3

        return imgcropped, intrinsic_cropped

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()

        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        rndseed = int(time.time())

        # RGB
        inputs['color'] = self.get_color(folder, frame_index, side, do_flip)

        # Depth Map
        inputs['depthgt'] = self.get_depthgt(folder, frame_index, side, do_flip)

        # intrinsic parameter
        inputs['K'] = self.get_intrinsic(folder, side)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug, rndseed)

        inputs['tag'] = self.initTag(index, do_flip)
        return inputs

    def initTag(self, index, do_flip):
        # Store Info of this Entry
        comps = self.filenames[index].split(' ')
        tag = str(comps[0] + ' ' + comps[1].zfill(6))
        if do_flip:
            tag = tag + ' flipYes'
        else:
            tag = tag + ' flipNoo'
        return tag

    def get_color(self, folder, frame_index, side, do_flip):
        color = pil.open(os.path.join(self.data_path, folder, self.dirmapping[side], "data/{}.png".format(str(frame_index).zfill(10))))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depthgt(self, folder, frame_index, side, do_flip):
        depthgt = pil.open(os.path.join(self.gt_path, folder, self.dirmapping[side], "{}.png".format(str(frame_index).zfill(10))))
        if do_flip:
            depthgt = depthgt.transpose(Image.FLIP_LEFT_RIGHT)
        return depthgt

    def get_intrinsic(self, folder, side):
        cam2cam = read_calib_file(os.path.join(self.data_path, folder.split('/')[0], 'calib_cam_to_cam.txt'))
        K = np.eye(4)
        K[0:3, :] = cam2cam['P_rect_0{}'.format(self.dirmapping[side][-1])].reshape(3, 4)
        return K