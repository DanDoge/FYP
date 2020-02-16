from .base_model import BaseModel
from .networks import _cal_kl, GANLoss, cat_feature
from util.image_pool import ImagePool
import itertools
import torch
import torchvision.transforms as transforms
from .networks_3d import _calc_grad_penalty
from .networks import *
from .basics import init_net
import numpy as np


class ContentStyleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_z', type=float, default=1.0, help='weight for ||E(G(random_z)) - random_z||')
        parser.add_argument('--lambda_kl_real', type=float, default=0.001, help='weight for KL loss, real')
        parser.add_argument('--lambda_mask', type=float, default=2.5, help='mask loss')
        parser.add_argument('--lambda_cycle_A', type=float, default=10.0, help='weight for forward cycle')
        parser.add_argument('--lambda_cycle_B', type=float, default=25.0, help='weight for backward cycle')
        parser.add_argument('--lambda_content_code', type=float, default=5.0, help='weight for forward cycle')
        parser.add_argument('--lambda_style_code', type=float, default=10.0, help='weight for backward cycle')
        return parser

    def __init__(self, opt, base_init=True):
        assert opt.input_nc == 1 and opt.output_nc == 3
        if base_init:
            BaseModel.__init__(self, opt)
        self.nz_texture = opt.nz_texture
        self.use_df = opt.use_df or opt.dataset_mode.find('df') >= 0
        self.vae = opt.lambda_kl_real > 0.0
        self.bg_B = -1
        self.bg_A = 1
        self.n_downsample = 4
        self.content_dim = 16
        self.n_res = 4
        self.vp_dim = 2

        if self.isTrain:
            self.model_names += ['E_style', 'G_real', 'G_depth', 'D_real', 'D_depth']
        else:
            self.model_names += ['E_style', 'G_real', 'G_depth']

        self.visual_names += ['real_A', 'real_B', 'rec_A', 'rec_B', 'rec_Aref', 'rec_Bref', 'fake_A', 'fake_B', 'fake_Brandom']
        self.loss_names += ['G_A', 'G_B', 'cycle_A', 'cycle_B', 'D_depth', 'D_real']
        self.cuda_names += ['z_texture']

        if opt.lambda_kl_real > 0.0:
            self.loss_names += ['kl_real', 'mu_enc', 'var_enc']


        # load/define networks: define G
        self.netG_real = self.define_G(opt.input_nc + self.vp_dim, opt.output_nc, opt.nz_texture, ext='AB')
        self.netG_depth = self.define_G(opt.output_nc + self.vp_dim, opt.input_nc, self.vp_dim, ext='BA')
        self.netE_style = self.define_E(opt.output_nc + self.vp_dim, self.vae)

        if opt.isTrain:
            self.netD_real = self.define_D(opt.output_nc, ext='A')
            self.netD_depth = self.define_D(opt.input_nc, ext='B')


        self.critCycle = torch.nn.L1Loss().to(self.device)
        self.critGAN = GANLoss(gan_mode=opt.gan_mode).to(self.device)

        self.optimizer_E = torch.optim.Adam(itertools.chain(self.netE_style.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_real.parameters(), self.netG_depth.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_depth.parameters(), self.netD_real.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers += [self.optimizer_E, self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        self.mask_A = input['Am'].to(self.device)
        self.mask_B = input['Bm'].to(self.device)
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_A2B = input['Ar'].to(self.device)

        self.mask_Aref = input['Amref'].to(self.device)
        self.mask_Bref = input['Bmref'].to(self.device)
        self.real_Aref = input['Aref'].to(self.device)
        self.real_Bref = input['Bref'].to(self.device)

        def local_tsfm(vp):
            return (vp.float() - torch.Tensor([180., 0.])) / torch.Tensor([180., 90.])

        self.vp_A = local_tsfm(input["Avp"]).to(self.device)
        self.vp_Aref = local_tsfm(input["Arefvp"]).to(self.device)
        self.vp_B = local_tsfm(input["Bvp"]).to(self.device)
        self.vp_Bref = local_tsfm(input["Brefvp"]).to(self.device)

        self.bs = self.real_B.size(0)
        self.z_texture = self.get_z_random(self.bs, self.nz_texture)

    def encode_style(self, input_image, vae=False):
        if vae:
            mu, logvar = self.netE_style(input_image)
            std = logvar.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
            return eps.mul(std).add_(mu), mu, logvar
        else:
            z = self.netE_style(input_image)
            return z, None, None

    def backward_GE(self, is_test=False):
        realB_with_vp = cat_feature(self.real_B, self.vp_B)
        self.z_style_B, mu_style_B, logvar_style_B = self.encode_style(realB_with_vp, self.vae)

        if is_test:
            self.z_style_B = mu_style_B

        self.fake_A = self.apply_mask(self.netG_depth(realB_with_vp, self.vp_B), self.mask_B, self.bg_A)
        self.fake_Aref = self.apply_mask(self.netG_depth(realB_with_vp, self.vp_Bref), self.mask_Bref, self.bg_A)
        self.loss_G_A = self.critGAN(self.netD_depth(cat_feature(self.fake_A, self.vp_B)), True) + self.critGAN(self.netD_depth(cat_feature(self.fake_Aref, self.vp_Bref)), True)

        self.rec_B = self.apply_mask(self.netG_real(self.fake_A, self.z_style_B), self.mask_B, self.bg_B)
        self.rec_Bref = self.apply_mask(self.netG_real(self.fake_Aref, self.z_style_B), self.mask_Bref, self.bg_B)
        self.loss_cycle_B = (self.critCycle(self.real_Bref, self.rec_Bref)+ self.critCycle(self.real_B, self.rec_B)) * self.opt.lambda_cycle_B

        realA_with_vp = cat_feature(self.real_A, self.vp_A)
        realAref_with_vp = cat_feature(self.real_Aref, self.vp_Aref)

        self.fake_B = self.apply_mask(self.netG_real(realA_with_vp, self.z_style_B), self.mask_A, self.bg_B)
        self.fake_Bref = self.apply_mask(self.netG_real(realAref_with_vp, self.z_style_B), self.mask_Aref, self.bg_B)
        self.loss_G_B = self.critGAN(self.netD_real(cat_feature(self.fake_B, self.vp_A)), True) + self.critGAN(self.netD_real(cat_feature(self.fake_Bref, self.vp_Aref)), True)

        self.rec_A = self.apply_mask(self.netG_depth(cat_feature(self.fake_B, self.vp_A), self.vp_A), self.mask_A, self.bg_A)
        self.rec_Aref = self.apply_mask(self.netG_depth(cat_feature(self.fake_Bref, self.vp_Aref), self.vp_Aref), self.mask_Aref, self.bg_A)
        self.loss_cycle_A = (self.critCycle(self.real_Aref, self.rec_Aref) + self.critCycle(self.real_A, self.rec_A)) * self.opt.lambda_cycle_A

        self.fake_Brandom = self.apply_mask(self.netG_real(realA_with_vp, self.z_texture), self.mask_A, self.bg_B)


        if self.opt.lambda_kl_real > 0.0:
            self.loss_mu_enc = torch.mean(torch.abs(mu_style_B))
            self.loss_var_enc = torch.mean(logvar_style_B.exp())
            self.loss_kl_real = _cal_kl(mu_style_B, logvar_style_B, self.opt.lambda_kl_real)
        # combined loss
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B \
                        + self.loss_G_A + self.loss_G_B \
                        + self.loss_mu_enc + self.loss_var_enc + self.loss_kl_real \
        self.loss_G.backward()

    def backward_D(self, epoch):
        '''
        if epoch > 100:
            real_A = self.real_A
        else:
            real_A = self.rec_A.detach() * (100 - epoch) / 100 + self.real_A.detach() * epoch / 100
        '''
        real_A = self.rec_A.detach()
        loss_D_depth_real = self.critGAN(self.netD_depth(cat_feature(real_A, self.vp_A)), True)
        loss_D_depth_fake = self.critGAN(self.netD_depth(cat_feature(self.fake_A.detach(), self.vp_B)), False)
        self.loss_D_depth = loss_D_depth_real + loss_D_depth_fake


        '''
        if epoch > 100:
            real_B = self.real_B
        else:
            real_B = self.rec_B.detach() * (100 - epoch) / 100 + self.real_B.detach() * epoch / 100
        '''
        real_B = self.rec_B.detach()
        loss_D_real_real = self.critGAN(self.netD_real(cat_feature(real_B, self.vp_B)), True)
        loss_D_real_fake = self.critGAN(self.netD_real(cat_feature(self.fake_B.detach(), self.vp_A)), False)
        self.loss_D_real = loss_D_real_real + loss_D_real_fake
        self.loss_D = self.loss_D_real + self.loss_D_depth
        self.loss_D.backward()


    def update_D(self, epoch=0):
        self.set_requires_grad([self.netD_depth, self.netD_real], True)
        self.optimizer_D.zero_grad()
        self.backward_D(epoch)
        self.optimizer_D.step()

    def update_G(self, epoch=0):
        self.set_requires_grad([self.netD_depth, self.netD_real], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_GE()
        self.optimizer_E.step()
        self.optimizer_G.step()
