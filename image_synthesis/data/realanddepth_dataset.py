import os.path
from data.base_dataset import BaseDataset, get_transform, get_normaliztion
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import torch
from torch.nn.functional import pad as pad_tensor
from os.path import join, dirname


class Depth2RealDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        root = '/data1/huangdj/PyTorch-CycleGAN-master/data'
        real_root = '/data1/huangdj/PyTorch-CycleGAN-master/real_data'
        self.transform_mask = get_transform(opt, has_mask=True, no_flip=True, no_normalize=True)
        self.transform_rgb = get_transform(opt, has_mask=False, no_flip=True, no_normalize=True)

        self.model_list = self.get_models(root)
        self.model2image = {}
        self.len_depth = 0
        self.len_albedo = 0
        self.get_AB(root)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--random_shift', action='store_true', help='add random shift to real images and rendered ones')
        parser.add_argument('--color_jitter', action='store_false', help='jitter the hue of loaded images')
        # type of  pose pool to sample from:
        parser.add_argument('--pose_type', type=str, default='hack', choices=['hack'], help='select which pool of poses to sample from')
        parser.add_argument('--pose_align', action='store_true', help='choose to shuffle pose or not. not shuffling == paired pose')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

        return parser


    def get_models(self, root):
        return os.listdir(root)

    def get_AB(self, root):
        self.len_depth = 0
        self.len_albedo = len(os.listdir(real_root))
        for model in self.model_list:
            self.model2image[model] = {"depth": [], "albedo": []}
            if model.endswith(".tar"):
                continue
            path_model = os.path.join(root, model)
            for img in os.listdir(path_model):
                if "depth" in img:
                    self.model2image[model]["depth"].append(os.path.join(path_model, img))
                    self.len_depth += 1


    def __getitem__(self, index):
        model_A = self.model_list[random.randint(0, len(self.model_list) - 1)]
        fileA = self.model2image[model_A]["depth"][random.randint(0, len(self.model2image[model_A]["depth"]) - 1)]
        fileAreal = fileA.replace("depth", "albedo")
        fileA_vp = torch.Tensor([int(fileA.split("_")[-4]), int(fileA.split("_")[-2])])

        fileAref = self.model2image[model_A]["depth"][random.randint(0, len(self.model2image[model_A]["depth"]) - 1)]
        fileAref_vp = torch.Tensor([int(fileAref.split("_")[-4]), int(fileAref.split("_")[-2])])

        fileB = self.real_root + '/' + str(random.randint(0, self.len_albedo - 1)) + '.png'
        fileBvp = torch.Tensor([0, 0])

        def get_depth_image(file_depth):
            img_depth = Image.open(file_depth)
            rgb_depth = img_depth.convert("L")
            rgb_depth = self.transform_rgb(rgb_depth)
            mask_depth = torch.zeros_like(rgb_depth)
            mask_depth[rgb_depth != 1.] = 1
            rgb_depth = (rgb_depth - 0.5) / 0.5
            return rgb_depth, mask_depth

        def get_rgb_image(file_rgb):
            img_rgb = Image.open(file_rgb)
            rgb_rgb = img_rgb.convert("RGB")
            rgb_rgb = self.transform_rgb(rgb_rgb)
            rgb_rgb = get_normaliztion()(rgb_rgb)
            mask_rgb = torch.zeros_like(rgb_rgb)
            img_maskB_rgb = img_rgb.convert("L")
            mask_rgb[img_maskB_rgb != 1.] = 1
            mask_rgb = mask_rgb.unsqueeze(0)
            return rgb_rgb, mask_rgb

        rgbA, maskA = get_depth_image(fileA)

        rgbB, maskB = get_rgb_image(fileB)


        return {'A': rgbA, 'B': rgbB, 'Am': maskA, 'Bm': maskB, "Avp": fileA_vp, "Bvp": fileB_vp}

    def __len__(self):
        return max(self.len_depth, self.len_albedo)
