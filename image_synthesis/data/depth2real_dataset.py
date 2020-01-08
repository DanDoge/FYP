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
        parser.add_argument('--color_jitter', action='store_true', help='jitter the hue of loaded images')
        # type of  pose pool to sample from:
        parser.add_argument('--pose_type', type=str, default='hack', choices=['hack'], help='select which pool of poses to sample from')
        parser.add_argument('--pose_align', action='store_true', help='choose to shuffle pose or not. not shuffling == paired pose')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

        return parser


    def get_models(self, root):
        return os.listdir(root)

    def get_AB(self, root):
        self.len_depth = 0
        self.len_albedo = 0
        for model in self.model_list:
            self.model2image[model] = {"depth": [], "albedo": []}
            if model.endswith(".tar"):
                continue
            path_model = os.path.join(root, model)
            for img in os.listdir(path_model):
                if "depth" in img:
                    self.model2image[model]["depth"].append(os.path.join(path_model, img))
                    self.len_depth += 1
                if "albedo" in img:
                    self.model2image[model]["albedo"].append(os.path.join(path_model, img))
                    self.len_albedo += 1


    def __getitem__(self, index):
        model_A = self.model_list[random.randint(0, len(self.model_list) - 1)]
        fileA = self.model2image[model_A]["depth"][random.randint(0, len(self.model2image[model_A]["depth"]) - 1)]
        fileAreal = fileA.replace("depth", "albedo")
        fileA_vp = torch.Tensor([int(fileA.split("_")[-4]), int(fileA.split("_")[-2])])

        fileAref = self.model2image[model_A]["depth"][random.randint(0, len(self.model2image[model_A]["depth"]) - 1)]
        fileAref_vp = torch.Tensor([int(fileAref.split("_")[-4]), int(fileAref.split("_")[-2])])

        model_B = self.model_list[random.randint(0, len(self.model_list) - 1)]
        fileB = self.model2image[model_B]["albedo"][random.randint(0, len(self.model2image[model_B]["albedo"]) - 1)]
        fileB_vp = torch.Tensor([int(fileB.split("_")[-4]), int(fileB.split("_")[-2])])

        fileBref = self.model2image[model_B]["albedo"][random.randint(0, len(self.model2image[model_B]["albedo"]) - 1)]
        fileBref_vp = torch.Tensor([int(fileBref.split("_")[-4]), int(fileBref.split("_")[-2])])


        imgA = Image.open(fileA)
        rgbA = imgA.convert("L")
        rgbA = self.transform_rgb(rgbA)
        maskA = torch.zeros_like(rgbA)
        maskA[rgbA != 1.] = 1
        rgbA = (rgbA - 0.5) / 0.5

        imgAref = Image.open(fileAref)
        rgbAref = imgAref.convert("L")
        rgbAref = self.transform_rgb(rgbAref)
        maskAref = torch.zeros_like(rgbAref)
        maskAref[rgbAref != 1.] = 1
        rgbAref = (rgbAref - 0.5) / 0.5

        imgA_real = Image.open(fileAreal)
        imgA_real = imgA_real.convert("RGB")
        imgA_real = self.transform_rgb(imgA_real)
        imgA_real = get_normaliztion()(imgA_real)

        imgB = Image.open(fileB)
        img_maskB = self.transform_mask(imgB)
        maskB = img_maskB[3, :, :]
        maskB = maskB.unsqueeze(0)
        rgbB = imgB.convert("RGB")
        rgbB = self.transform_rgb(rgbB)
        rgbB = get_normaliztion()(rgbB)

        imgBref = Image.open(fileBref)
        img_maskBref = self.transform_mask(imgBref)
        maskBref = img_maskBref[3, :, :]
        maskBref = maskBref.unsqueeze(0)
        rgbBref = imgBref.convert("RGB")
        rgbBref = self.transform_rgb(rgbBref)
        rgbBref = get_normaliztion()(rgbBref)


        return {'A': rgbA, 'B': rgbB, 'Am': maskA, 'Bm': maskB, 'Ar': imgA_real, "Aref": rgbAref, "Amref": maskAref, "Bref": rgbBref, "Bmref": maskBref, "Avp": fileA_vp, "Arefvp": fileAref_vp, "Bvp": fileB_vp, "Brefvp": fileBref_vp}

    def __len__(self):
        return max(self.len_depth, self.len_albedo)
