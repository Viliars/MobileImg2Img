import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
import cv2


class FaceDataset(data.Dataset):
    '''
    # -----------------------------------------
    # dataset for BSRGAN
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(FaceDataset, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['patch_size'] if self.opt['patch_size'] else 512

        self.paths_HQ = util.get_image_paths(opt['dataroot_hq'])
        self.paths_LQ = util.get_image_paths(opt['dataroot_lq'])

        assert self.paths_HQ, 'Error: HQ path is empty.'
        assert self.paths_LQ, 'Error: LQ path is empty.'

    def __getitem__(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        HQ_path = self.paths_HQ[index]
        LQ_path = self.paths_LQ[index]

        img_HQ = util.imread_uint(HQ_path, self.n_channels)
        img_LQ = util.imread_uint(LQ_path, self.n_channels)

        img_HQ = cv2.resize(img_HQ, (512, 512), interpolation=cv2.INTER_AREA)
        img_LQ = cv2.resize(img_LQ, (512, 512), interpolation=cv2.INTER_AREA)

        img_HQ = util.uint2single(img_HQ)
        img_LQ = util.uint2single(img_LQ)
            
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_LQ, img_HQ = util.single2tensor3(img_LQ), util.single2tensor3(img_HQ)

        return {'lq': img_LQ, 'hq': img_HQ, 'L_path': LQ_path, 'H_path': HQ_path}

    def __len__(self):
        return len(self.paths_HQ)


if __name__ == '__main__':
    from utils import utils_image as util
    opt = {
        "n_channels": 3,
        "patch_size": 140,
        "dataroot_hq": "/home/viliar/Documents/FFHQ/Train/HQ",
        "dataroot_lq": "/home/viliar/Documents/FFHQ/Train/LQ",
        "phase": "train"
        }

    dataset = FaceDataset(opt)
    for i in range(10):
        bufer = dataset[59999]
        lq = bufer["lq"]
        hq = bufer["hq"]

        img_lq = util.tensor2uint(lq)
        img_hq = util.tensor2uint(hq)

        img_concat = np.concatenate([img_lq, img_hq], axis=1)
        util.imsave(img_concat, f'FUCK/{i:03d}.png')
