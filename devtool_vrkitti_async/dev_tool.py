"""
    Sample Run:
    python dev_tool.py

    Calls the dataloader and plots the points obtained by flow and lidars.
    TODO: Check for occlusions on static and moving objects
"""

from __future__ import print_function, division
import os, inspect
project_rootdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import copy
from dataset_VRKitti2 import VirtualKITTI2

import torch.utils.data as data
import PIL.Image as Image

def vls_ins(rgb, anno):
    rgbc = copy.deepcopy(rgb)
    r = rgbc[:, :, 0].astype(np.float)
    g = rgbc[:, :, 1].astype(np.float)
    b = rgbc[:, :, 2].astype(np.float)
    for i in np.unique(anno):
        if i > 0:
            rndc = np.random.randint(0, 255, 3).astype(np.float)
            selector = anno == i
            r[selector] = rndc[0] * 0.25 + r[selector] * 0.75
            g[selector] = rndc[1] * 0.25 + g[selector] * 0.75
            b[selector] = rndc[2] * 0.25 + b[selector] * 0.75
    rgbvls = np.stack([r, g, b], axis=2)
    rgbvls = np.clip(rgbvls, a_max=255, a_min=0).astype(np.uint8)
    return Image.fromarray(rgbvls)

def tensor2disp(tensor, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('magma')
    tnp = tensor[viewind, 0, :, :].detach().cpu().numpy()
    if percentile is not None:
        vmax = np.percentile(tnp, percentile)
    tnp = tnp / vmax
    tnp = (cm(tnp) * 255).astype(np.uint8)
    return Image.fromarray(tnp[:, :, 0:3])

def read_splits():
    split_root = os.path.join(project_rootdir, 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'training_split.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'evaluation_split.txt'), 'r')]
    return train_entries, evaluation_entries

def vls_vrkitti2():
    # Instance map index from 0 to a max instance number where 0 always indicates background. If number of
    # instance is more than max instance number, it will be assigned to background

    # Poses contain pose to project frame 1 3D points in Image Coordinate system to frame 2. In term of obj
    # poses, it is a joint poses containing self movment as well.
    
    train_entries, evaluation_entries = read_splits()
    train_dataset = VirtualKITTI2(args=args, root=args.dataset_root, entries=train_entries)
    train_loader = data.DataLoader(train_dataset, batch_size=1, pin_memory=False, drop_last=True)
    for i_batch, data_blob in enumerate(train_loader):
        image1 = data_blob['img1']
        image2 = data_blob['img2']
        flowgt = data_blob['flowmap']
        depthgt = data_blob['depthmap']
        insmap = data_blob['insmap']

        intrinsic = data_blob['intrinsic']
        poses = data_blob['poses']

        img1 = image1[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)
        img2 = image2[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)

        depthgtnp = depthgt[0].squeeze().cpu().numpy()
        flownp = flowgt[0].squeeze().cpu().numpy()
        insnp = insmap[0].squeeze().cpu().numpy()
        posesnp = poses[0].cpu().numpy()
        intrinsicnp = intrinsic[0].cpu().numpy()

        bz, _, h, w = image1.shape
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        selector = (depthgtnp > 0) * (depthgtnp < 30)

        xxf = xx[selector]
        yyf = yy[selector]
        rndidx = np.random.randint(0, np.sum(selector), 1).item()
        rndx = xxf[rndidx]
        rndy = yyf[rndidx]
        rndd = depthgtnp[rndy, rndx]

        cm = plt.get_cmap('magma')
        rndcolor = cm([1 / rndd / 0.15])[:, 0:3]

        flowgtx = flownp[0, rndy, rndx]
        flowgty = flownp[1, rndy, rndx]

        # Compute Corresponding point at frame 2 according to Flowgt
        pts_frm2_fflow_x = rndx + flowgtx
        pts_frm2_fflow_y = rndy + flowgty

        # Compute Corresponding point at frame 2 according to Pose and Depth
        rndins = insnp[rndy, rndx]
        relpose = posesnp[rndins]
        projM = intrinsicnp @ relpose @ np.linalg.inv(intrinsicnp)
        pts3d = np.array([[rndx * rndd, rndy * rndd, rndd, 1]]).T
        pts2d = projM @ pts3d
        pts2d_d = pts2d[2, 0]
        pts2d_x = pts2d[0, 0] / pts2d_d
        pts2d_y = pts2d[1, 0] / pts2d_d

        # Compute Epipole of rnd instance on frame 2
        relpose = posesnp[rndins]
        projM = intrinsicnp @ relpose @ np.linalg.inv(intrinsicnp)
        pts3d_epp = np.array([[0 * rndd, 0 * rndd, 0, 1]]).T
        pts2d_epp = projM @ pts3d_epp
        pts2d_d_epp = pts2d_epp[2, 0]
        pts2d_x_epp = pts2d_epp[0, 0] / pts2d_d_epp
        pts2d_y_epp = pts2d_epp[1, 0] / pts2d_d_epp

        marker_size = 50
        fig = plt.figure(figsize=(16, 9))
        fig.add_subplot(2, 2, 1)
        plt.scatter(rndx, rndy, marker_size, 'r')
        plt.imshow(img1)
        plt.legend(['Sampled pts'])
        plt.title("Image at Frame T")

        fig.add_subplot(2, 2, 2)
        plt.imshow(img2)
        # Plot the old point as well
        plt.scatter(rndx, rndy, marker_size, 'r')
        # Plot the point obtained from flow
        plt.scatter(pts_frm2_fflow_x, pts_frm2_fflow_y, marker_size, 'b', alpha= 0.4)
        plt.scatter(pts2d_x, pts2d_y, marker_size, 'g')
        # Plot epipoles
        plt.scatter(pts2d_x_epp, pts2d_y_epp, marker_size, 'c')
        plt.legend(['Cor pts from flowgt', 'Cor pts from depthgt and posegt', 'Epipole'])
        plt.title("Image at Frame T + 1")

        fig.add_subplot(2, 2, 3)
        plt.imshow(tensor2disp(1/depthgt, vmax=0.15, viewind=0))
        plt.title("Disparity Image at Frame T")

        fig.add_subplot(2, 2, 4)
        plt.imshow(vls_ins(img1, insnp))
        plt.title("Instance Image at Frame T")

        plt.show()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="data/virtual_kitti_organized")
    parser.add_argument('--maxinsnum', type=int, default=20)

    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    vls_vrkitti2()