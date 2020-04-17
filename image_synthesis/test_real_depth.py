# import torch.backends.cudnn as cudnn
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
from PIL import Image
import numpy as np

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
opt.num_threads = 0
opt.serial_batches = True  # no shuffle
opt.batch_size = 1  # force to be 1
dataset = create_dataset(opt)
dataset_size = len(dataset)
print('#training data = %d' % dataset_size)
model = create_model(opt)
model.setup(opt)
total_steps = 0

model.netG_real.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/real_depth/car_df/2020-04-09/car_df/latest_net_G_real.pth"))
model.netG_depth.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/real_depth/car_df/2020-04-09/car_df/latest_net_G_depth.pth"))
model.netE_style.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/real_depth/car_df/2020-04-09/car_df/latest_net_E_style.pth"))

model.netG_real.eval()
model.netG_depth.eval()
model.netE_style.eval()

vpB = []
vpBref = []

style_synthesized = []
style_real = []
content_synthesized = []
vp_synthesized = []
mask_synthesized = []
bg_synthesized = []

def savefig(tensor, name, i):
    out_np = tensor[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
    im = Image.fromarray((out_np * 255).astype(np.uint8))
    im.save("./out/" + name + "/" + str(i) + ".jpg")
    return

def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat

def content_style_swap(model, content, syn_style, real_style, vp, mask, bg):
    for i in range(100):
        for j in range(100):
            syn_rec = model.apply_mask(model.netG_real(content[i], vp[i], syn_style[j]), mask[i], bg[i])
            real_rec = model.apply_mask(model.netG_real(content[i], vp[i], real_style[j]), mask[i], bg[i])
            savefig(syn_rec, "syn_swap", i * 100 + j)
            savefig(real_rec, "real_swap", i * 100 + j)

        for j in range(100):
            mix_syn_style = syn_style[i] * j / 100 + syn_style[int((i + 1) % 100)] * (100 - j) / 100
            mix_real_style = real_style[i] * j / 100 + real_style[int((i + 1) % 100)] * (100 - j) / 100
            syn_rec = model.apply_mask(model.netG_real(content[i], vp[i], mix_syn_style), mask[i], bg[i])
            real_rec = model.apply_mask(model.netG_real(content[i], vp[i], mix_real_style), mask[i], bg[i])
            savefig(syn_rec, "syn_interpolate", i * 100 + j)
            savefig(real_rec, "real_interpolate", i * 100 + j)
    return

def rotate(model, i):
    def local_tsfm(vp):
        return (vp.float() - torch.Tensor([180., 0.])) / torch.Tensor([180., 90.])

    for az in range(0, 360, 20):
        vp = torch.Tensor([az, 0])
        vp = local_tsfm(vp)
        fake_A = model.apply_mask(model.netG_depth(model.real_B, model.vp_B, vp), model.mask_B, model.bg_A)
        rec_B = model.apply_mask(model.netG_real(fake_A, vp, model.z_style_B), model.mask_B, model.bg_B)
        savefig(rec_B, "rotate", i * 100 + int(az / 20))



for epoch in range(1):

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - opt.epoch_count)
        model.set_input(data)
        if model.skip():
            continue

        savefig(model.real_B, "real", i)
        savefig(model.real_Bref, "realref", i)
        realB_with_vp = cat_feature(model.real_B, model.vp_B)
        model.z_style_B, mu_style_B, logvar_style_B = model.encode_style(realB_with_vp, model.vae)

        #model.fake_A = model.apply_mask(model.netG_depth(model.real_B, model.vp_B, model.vp_B), model.mask_B, model.bg_A)
        #model.fake_Aref = model.apply_mask(model.netG_depth(model.real_B, model.vp_B, model.vp_Bref), model.mask_Bref, model.bg_A)

        model.rec_B = model.apply_mask(model.netG_real(model.real_B2A, model.vp_B, model.z_style_B), model.mask_B, model.bg_B)
        savefig(model.rec_B, "rec", i)
        model.rec_Bref = model.apply_mask(model.netG_real(model.real_Bref2A, model.vp_Bref, model.z_style_B), model.mask_Bref, model.bg_B)
        savefig(model.rec_Bref, "novelview", i)

        model.fake_B = model.apply_mask(model.netG_real(model.real_A, model.vp_A, model.z_style_B), model.mask_A, model.bg_B)
        savefig(model.fake_B, "fake", i)

        model.fake_Brandom = model.apply_mask(model.netG_real(model.real_A, model.vp_A, model.z_texture), model.mask_A, model.bg_B)
        savefig(model.fake_Brandom, "random", i)

        realBreal_with_vp = cat_feature(model.real_Breal, model.vp_Breal)
        model.z_style_Breal, mu_style_Breal, logvar_style_Breal = model.encode_style(realBreal_with_vp, model.vae)
        model.fake_Breal = model.apply_mask(model.netG_real(model.real_A, model.vp_Breal, model.z_style_Breal), model.mask_A, model.bg_B)
        savefig(model.fake_Breal, "fake_real", i)
        savefig(model.real_Breal, "real_real", i)

        if i < 100:
            content_synthesized.append(model.real_B2A)
            style_synthesized.append(model.z_style_B)
            style_real.append(model.z_style_Breal)
            vp_synthesized.append(model.vp_B)
            mask_synthesized.append(model.mask_B)
            bg_synthesized.append(model.bg_B)
            rotate(model, i)
        else:
           break


        vpB.append(model.vp_B[0].detach().cpu().numpy())
        vpBref.append(model.vp_Bref[0].detach().cpu().numpy())

        if i > 10000:
            break

    import pickle
    pickle.dump(vpB, open("./out/vp_B", "wb"))
    pickle.dump(vpBref, open("./out/vp_Bref", "wb"))

    content_style_swap(model, content_synthesized, style_synthesized, style_real, vp_synthesized, mask_synthesized, bg_synthesized)
    break
