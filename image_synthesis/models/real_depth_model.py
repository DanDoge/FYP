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
import random
#from .roi_align.functions import roi_align


class RealDepthModel(BaseModel):
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
            self.model_names += ['E_style', 'G_real', 'G_depth', 'D_real', 'D_depth', 'D_real_single', 'D_real_local']
        else:
            self.model_names += ['E_style', 'G_real', 'G_depth']

        self.visual_names += ['real_A', 'real_B', 'rec_Aref', 'rec_Bref', 'fake_Aref', 'fake_Bref', 'fake_Breal']
        self.loss_names += ['G_A', 'G_B', 'cycle_A', 'cycle_B', 'D_depth', 'D_real', 'D_local']
        self.cuda_names += ['z_texture']

        if opt.lambda_kl_real > 0.0:
            self.loss_names += ['kl_real', 'mu_enc', 'var_enc']


        # load/define networks: define G
        self.netG_real = self.define_G(opt.input_nc, opt.output_nc, opt.nz_texture, num_downs=2, ext='AB', ngf=opt.ngf*2)
        self.netG_depth = self.define_G(opt.output_nc, opt.input_nc, self.vp_dim, num_downs=self.n_downsample, ext='BA', ngf=opt.ngf)
        self.netE_style = self.define_E(opt.output_nc + self.vp_dim, self.vae)

        if opt.isTrain:
            self.netD_real = self.define_D(opt.output_nc * 2, ext='A')
            self.netD_depth = self.define_D(opt.input_nc * 2, ext='B')
            self.netD_real_single = self.define_D(opt.output_nc, ext='Asingle')
            self.netD_depth_single = self.define_D(opt.input_nc, ext='Bsingle')
            self.netD_real_local = self.define_D(opt.output_nc * 2, ext='Blocal')


        self.critCycle = torch.nn.L1Loss().to(self.device)
        self.critGAN = GANLoss(gan_mode=opt.gan_mode).to(self.device)

        self.optimizer_E = torch.optim.Adam(itertools.chain(self.netE_style.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_real.parameters(), self.netG_depth.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_depth.parameters(), self.netD_real.parameters(), self.netD_depth_single.parameters(), self.netD_real_single.parameters(), self.netD_real_local.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers += [self.optimizer_E, self.optimizer_G, self.optimizer_D]
    def set_input(self, input):
        self.mask_A = input['Am'].to(self.device)
        self.mask_B = input['Bm'].to(self.device)
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.blur_B = input['Bblur'].to(self.device)
        self.real_A2B = input['Ar'].to(self.device)
        self.real_B2A = input["Bd"].to(self.device)
        self.real_Bref2A = input["Brefd"].to(self.device)

        self.real_Breal = input["Breal"].to(self.device)
        self.mask_Breal = input["Brealm"].to(self.device)

        self.mask_Aref = input['Amref'].to(self.device)
        self.mask_Bref = input['Bmref'].to(self.device)
        self.real_Aref = input['Aref'].to(self.device)
        self.real_Bref = input['Bref'].to(self.device)

        self.real_A = self.apply_mask(self.real_A, self.mask_A, self.bg_A)
        self.real_B = self.apply_mask(self.real_B, self.mask_B, self.bg_B)
        self.blur_B = self.apply_mask(self.blur_B, self.mask_B, self.bg_B)
        self.real_A2B = self.apply_mask(self.real_A2B, self.mask_A, self.bg_B)
        self.real_B2A = self.apply_mask(self.real_B2A, self.mask_B, self.bg_A)
        self.real_Bref2A = self.apply_mask(self.real_Bref2A, self.mask_Bref, self.bg_A)
        self.real_Aref = self.apply_mask(self.real_Aref, self.mask_Aref, self.bg_A)
        self.real_Bref = self.apply_mask(self.real_Bref, self.mask_Bref, self.bg_B)

        def local_tsfm(vp):
            return (vp.float() - torch.Tensor([180., 0.])) / torch.Tensor([180., 90.])

        self.vp_A = local_tsfm(input["Avp"]).to(self.device)
        self.vp_Aref = local_tsfm(input["Arefvp"]).to(self.device)
        self.vp_B = local_tsfm(input["Bvp"]).to(self.device)
        self.vp_Bref = local_tsfm(input["Brefvp"]).to(self.device)

        self.vp_Breal = local_tsfm(input["Brealvp"]).to(self.device)

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

        realBref_with_vp = cat_feature(self.real_Bref, self.vp_Bref)
        self.z_style_Bref, mu_style_Bref, logvar_style_Bref = self.encode_style(realBref_with_vp, self.vae)

        #print(self.real_B.shape, self.z_style_B.shape)



        self.fake_A = self.apply_mask(self.netG_depth(self.real_B, self.vp_B, self.vp_B), self.mask_B, self.bg_A)
        self.fake_Aref = self.apply_mask(self.netG_depth(self.real_B, self.vp_B, self.vp_Bref), self.mask_Bref, self.bg_A)


        self.fake_B = self.apply_mask(self.netG_real(self.real_A, self.vp_A, self.z_style_B), self.mask_A, self.bg_B)
        self.fake_Bref = self.apply_mask(self.netG_real(self.real_Aref, self.vp_Aref, self.z_style_B), self.mask_Aref, self.bg_B)
        self.rec_A = self.apply_mask(self.netG_depth(self.fake_B, self.vp_A, self.vp_A), self.mask_A, self.bg_A)
        self.rec_Aref = self.apply_mask(self.netG_depth(self.fake_Bref, self.vp_Aref, self.vp_Aref), self.mask_Aref, self.bg_A)
        self.loss_cycle_A = (self.critCycle(self.real_Aref, self.rec_Aref) + self.critCycle(self.real_A, self.rec_A)) * self.opt.lambda_cycle_A

        self.loss_G_A = self.critGAN(self.netD_depth(torch.cat([self.fake_A, self.real_Bref2A], 1)), True) \
                        + self.critGAN(self.netD_depth(torch.cat([self.fake_Aref, self.real_B2A], 1)), True) \
                        + self.critGAN(self.netD_depth(torch.cat([self.rec_A, self.real_Aref], 1)), True) \
                        + self.critGAN(self.netD_depth(torch.cat([self.rec_Aref, self.real_A], 1)), True) \
                        + self.critGAN(self.netD_depth_single(self.fake_A), True) \
                        + self.critGAN(self.netD_depth_single(self.fake_Aref), True) \
                        + self.critGAN(self.netD_depth_single(self.rec_A), True) \
                        + self.critGAN(self.netD_depth_single(self.rec_Aref), True) \
                        + self.critCycle(self.fake_A, self.real_B2A) * self.opt.lambda_cycle_A \
                        + self.critCycle(self.fake_Aref, self.real_Bref2A) * self.opt.lambda_cycle_A


        self.rec_B = self.apply_mask(self.netG_real(self.fake_A, self.vp_B, self.z_style_B), self.mask_B, self.bg_B)
        self.rec_Bref = self.apply_mask(self.netG_real(self.fake_Aref, self.vp_Bref, self.z_style_B), self.mask_Bref, self.bg_B)
        self.loss_cycle_B = (self.critCycle(self.real_Bref, self.rec_Bref)+ self.critCycle(self.real_B, self.rec_B)) * self.opt.lambda_cycle_B


        realBreal_with_vp = cat_feature(self.real_Breal, self.vp_Breal)
        self.z_style_Breal, mu_style_Breal, logvar_style_Breal = self.encode_style(realBreal_with_vp, self.vae)
        self.fake_Breal = self.apply_mask(self.netG_real(self.real_A, self.vp_A, self.z_style_Breal), self.mask_A, self.bg_B)

        self.loss_G_B = self.critGAN(self.netD_real(torch.cat([self.fake_B, self.real_Bref], 1)), True) \
                        + self.critGAN(self.netD_real(torch.cat([self.fake_Bref, self.real_B], 1)), True) \
                        + self.critGAN(self.netD_real(torch.cat([self.rec_B, self.real_Bref], 1)), True) \
                        + self.critGAN(self.netD_real(torch.cat([self.rec_Bref, self.real_B], 1)), True) \
                        + self.critGAN(self.netD_real(torch.cat([self.real_Breal, self.fake_Breal], 1)), True) \
                        + self.critGAN(self.netD_real_single(self.fake_B), True) \
                        + self.critGAN(self.netD_real_single(self.fake_Bref), True) \
                        + self.critGAN(self.netD_real_single(self.rec_B), True) \
                        + self.critGAN(self.netD_real_single(self.rec_Bref), True) \
                        + self.critGAN(self.netD_real_single(self.fake_Breal), True)


        self.fake_Brandom = self.apply_mask(self.netG_real(self.real_A, self.vp_A, self.z_texture), self.mask_A, self.bg_B)
        fakeB_with_vp = cat_feature(self.fake_Brandom, self.vp_A)
        self.z_texture_rec, mu_style_random, logvar_style_random = self.encode_style(fakeB_with_vp, self.vae)


        self.loss_mu_enc = self.critCycle(self.z_style_Bref, self.z_style_B.detach()) + self.critCycle(self.z_texture, self.z_texture_rec) + self.critGAN(self.netD_real_single(self.fake_Brandom), True)
        if self.opt.lambda_kl_real > 0.0:
            self.loss_mu_enc += torch.mean(torch.abs(mu_style_B))
            self.loss_var_enc = torch.mean(logvar_style_B.exp())
            self.loss_kl_real = _cal_kl(mu_style_B, logvar_style_B, self.opt.lambda_kl_real)


        self.input_patch, self.ref_patch, self.target_patch, self.blur_patch = self.generate_random_block(self.real_B, self.real_Bref, self.fake_B, self.blur_B)
        self.real_pair = torch.cat([self.input_patch, self.ref_patch], 1)
        self.blur_pair = torch.cat([self.input_patch, self.blur_patch], 1)
        self.fake_pair=  torch.cat([self.input_patch, self.target_patch], 1)
        self.loss_D_local = self.critGAN(self.netD_real_local(self.fake_pair), True)
        # combined loss
        '''
        self.loss_cycle_B = 0.0
        self.loss_G_B = 0.0
        self.loss_mu_enc = 0.0
        self.loss_var_enc = 0.0
        self.loss_kl_real = 0.0
        '''
        self.loss_G = self.loss_cycle_A + self.loss_cycle_B \
                        + self.loss_G_A + self.loss_G_B \
                        + self.loss_mu_enc + self.loss_var_enc + self.loss_kl_real \
                        + self.loss_D_local
        self.loss_G.backward()

    def backward_D(self, epoch):

        loss_D_depth_real = self.critGAN(self.netD_depth(torch.cat([self.real_A.detach(), self.real_Aref.detach()], 1)), True) * 4 + self.critGAN(self.netD_depth_single(self.real_A.detach()), True) * 2 + self.critGAN(self.netD_depth_single(self.real_Aref.detach()), True) * 2
        loss_D_depth_fake = self.critGAN(self.netD_depth(torch.cat([self.fake_A.detach(), self.real_Bref2A.detach()], 1)), False) \
                        + self.critGAN(self.netD_depth(torch.cat([self.fake_Aref.detach(), self.real_B2A.detach()], 1)), False) \
                        + self.critGAN(self.netD_depth(torch.cat([self.rec_A.detach(), self.real_Aref.detach()], 1)), False) \
                        + self.critGAN(self.netD_depth(torch.cat([self.rec_Aref.detach(), self.real_A.detach()], 1)), False) \
                        + self.critGAN(self.netD_depth_single(self.fake_A.detach()), False) \
                        + self.critGAN(self.netD_depth_single(self.fake_Aref.detach()), False) \
                        + self.critGAN(self.netD_depth_single(self.rec_A.detach()), False) \
                        + self.critGAN(self.netD_depth_single(self.rec_Aref.detach()), False)
        self.loss_D_depth = loss_D_depth_real + loss_D_depth_fake




        loss_D_real_real = self.critGAN(self.netD_real(torch.cat([self.real_B.detach(), self.real_Bref.detach()], 1)), True) * 4 + self.critGAN(self.netD_real_single(self.real_B.detach()), True) * 2 + self.critGAN(self.netD_real_single(self.real_Bref.detach()), True) * 2
        loss_D_real_fake = self.critGAN(self.netD_real(torch.cat([self.rec_B.detach(), self.real_Bref.detach()], 1)), False) \
                           + self.critGAN(self.netD_real(torch.cat([self.rec_Bref.detach(), self.real_B.detach()], 1)), False) \
                           + self.critGAN(self.netD_real(torch.cat([self.fake_B.detach(), self.real_Bref.detach()], 1)), False) \
                           + self.critGAN(self.netD_real(torch.cat([self.fake_Bref.detach(), self.real_B.detach()], 1)), False) \
                           + self.critGAN(self.netD_real(torch.cat([self.real_Breal.detach(), self.fake_Breal.detach()], 1)), False) \
                           + self.critGAN(self.netD_real_single(self.fake_B.detach()), False) \
                           + self.critGAN(self.netD_real_single(self.fake_Bref.detach()), False) \
                           + self.critGAN(self.netD_real_single(self.rec_B.detach()), False) \
                           + self.critGAN(self.netD_real_single(self.rec_Bref.detach()), False) \
                           + self.critGAN(self.netD_real_single(self.fake_Brandom.detach()), False) \
                           + self.critGAN(self.netD_real_single(self.fake_Breal.detach()), False)

        loss_D_real_single = self.critGAN(self.netD_real_local(self.fake_pair.detach()), False) \
                             + self.critGAN(self.netD_real_local(self.real_pair.detach()), True) \
                             + self.critGAN(self.netD_real_local(self.blur_pair.detach()), False)
        self.loss_D_real = loss_D_real_real + loss_D_real_fake


        self.loss_D = self.loss_D_real + self.loss_D_depth
        #print("!!!")
        self.loss_D.backward()


    def update_D(self, epoch=0):
        self.set_requires_grad([self.netD_depth, self.netD_real, self.netD_depth_single, self.netD_real_single, self.netD_real_local], True)
        self.optimizer_D.zero_grad()
        self.backward_D(epoch)
        self.optimizer_D.step()

    def update_G(self, epoch=0):
        self.set_requires_grad([self.netD_depth, self.netD_real, self.netD_depth_single, self.netD_real_single, self.netD_real_local], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_GE()
        self.optimizer_E.step()
        self.optimizer_G.step()


    def generate_random_block(self, input, inputref, target, blurs):
        batch_size, channel, height, width = target.size()  # B X 3*nencode X 64 X 64
        target_tensor = target.data
        block_size = 64
        img_size = self.opt.crop_size  # 128 / 256

        if True:
            if True:
                rand_idx = 0
                x = random.randint(0, height - block_size - 1)
                y = random.randint(0, width - block_size - 1)
                target_random_block = torch.tensor(target_tensor[:, :, x:x + block_size, y:y + block_size], requires_grad=True)
                #rois = Variable(torch.FloatTensor([[i, x, y, x + block_size, y + block_size] for i in range(batch_size)]).to(self.device))
                #target_random_block = roi_align.roi_align_op(target_tensor, rois, (block_size, block_size), 1.)
                #print(target_random_block)
                x = random.randint(0, height - block_size - 1)
                y = random.randint(0, width - block_size - 1)
                target_blur_block = torch.tensor(blurs[:, :,x:x + block_size, y:y + block_size], requires_grad=True)


                target_blocks = target_random_block
                blur_blocks = target_blur_block

                x1 = random.randint(0, height - block_size - 1)
                y1 = random.randint(0, width - block_size - 1)
                input_random_block = torch.tensor(input[:, :, x1:x1 + block_size, y1:y1 + block_size], requires_grad=True)
                x1 = random.randint(0, height - block_size - 1)
                y1 = random.randint(0, width - block_size - 1)
                ref_random_block = torch.tensor(inputref[:, :, x1:x1 + block_size, y1:y1 + block_size], requires_grad=True)
                input_blocks = input_random_block
                ref_blocks = ref_random_block
                #print(input_blocks.size(), ref_blocks.size())

        return input_blocks, ref_blocks, target_blocks, blur_blocks
