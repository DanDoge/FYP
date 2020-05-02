# import torch.backends.cudnn as cudnn
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_transform, get_normaliztion


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

model.netG_real.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/real_depth/car_df/2020-04-09/car_df/1_net_G_real.pth"))
model.netG_depth.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/real_depth/car_df/2020-04-09/car_df/1_net_G_depth.pth"))
model.netE_style.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/real_depth/car_df/2020-04-09/car_df/1_net_E_style.pth"))

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

style_list = []
style_real_list = []

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
    def get_rgb_image(file_rgb):
        img_rgb = Image.open(file_rgb)
        img_mask_rgb = dataset.dataset.transform_mask(img_rgb)
        mask_rgb = img_mask_rgb[3, :, :]
        mask_rgb = mask_rgb.unsqueeze(0)

        rgb_rgb = img_rgb.convert("RGB")
        rgb_rgb = dataset.dataset.transform_rgb(rgb_rgb)
        rgb_rgb = get_normaliztion()(rgb_rgb)

        return rgb_rgb, mask_rgb

    for fileB in model.fileBlist:
        #print(fileB[0])
        vp = torch.Tensor([[int(fileB[0].split("_")[-4]), int(fileB[0].split("_")[-2])]])
        vp = local_tsfm(vp).to(model.device)
        #print(model.vp_B, vp)
        real_B, mask_B = get_rgb_image(fileB[0])
        real_B.to(model.device)
        mask_B.to(model.device)
        fake_A = model.apply_mask(model.netG_depth(model.real_B, model.vp_B, vp), mask_B, model.bg_A)
        rec_B = model.apply_mask(model.netG_real(fake_A, vp, model.z_style_B), mask_B, model.bg_B)
        savefig(rec_B, "rotate", i * 1000 + int(fileB[0].split("_")[-4]) / 20 * 10 + int(fileB[0].split("_")[-2]) / 10)

    if i < 10:
        minloss = 123456789.0
        viewsynthesis_mattix = []
        for srcfileB in model.fileBlist:
            loss = 0.0
            for tgtfileB in model.fileBlist:
            #print(fileB[0])
                srcvp = torch.Tensor([[int(srcfileB[0].split("_")[-4]), int(srcfileB[0].split("_")[-2])]])
                srcvp = local_tsfm(srcvp).to(model.device)
                tgtvp = torch.Tensor([[int(tgtfileB[0].split("_")[-4]), int(tgtfileB[0].split("_")[-2])]])
                tgtvp = local_tsfm(tgtvp).to(model.device)
                #print(model.vp_B, vp)
                srcreal_B, srcmask_B = get_rgb_image(srcfileB[0])
                srcreal_B.to(model.device)
                srcmask_B.to(model.device)
                tgtreal_B, tgtmask_B = get_rgb_image(tgtfileB[0])
                tgtreal_B.to(model.device)
                tgtmask_B.to(model.device)
                srcrealB_with_vp = cat_feature(srcreal_B, srcvp)
                srcz_style_B, mu_style_B, logvar_style_B = model.encode_style(srcrealB_with_vp, model.vae)
                fake_A = model.apply_mask(model.netG_depth(srcreal_B, srcvp, tgtvp), tgtmask_B, model.bg_A)
                rec_B = model.apply_mask(model.netG_real(fake_A, tgtvp, srcz_style_B), tgtmask_B, model.bg_B)
                loss = torch.mean(torch.abs(rec_B - tgtreal_B))
                viewsynthesis_mattix.append([srcfileB, tgtfileB, loss])
        import pickle
        with open("./out/vs_mat_" + str(i), "wb") as f:
            pickle.dump(viewsynthesis_mattix, f)



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

        style_list.append(model.z_style_B[0].detach().cpu().numpy())

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
        model.fake_Breal = model.apply_mask(model.netG_real(model.real_A, model.vp_B, model.z_style_Breal), model.mask_A, model.bg_B)
        savefig(model.fake_Breal, "fake_real", i)
        savefig(model.real_Breal, "real_real", i)

        style_real_list.append(model.z_style_Breal[0].detach().cpu().numpy())

        if i < 100:
            content_synthesized.append(model.real_B2A)
            style_synthesized.append(model.z_style_B)
            style_real.append(model.z_style_Breal)
            vp_synthesized.append(model.vp_B)
            mask_synthesized.append(model.mask_B)
            bg_synthesized.append(model.bg_B)
            rotate(model, i)
        #else:
           #break


        vpB.append(model.vp_B[0].detach().cpu().numpy())
        vpBref.append(model.vp_Bref[0].detach().cpu().numpy())

        if i > 10000:
            break

    import pickle
    pickle.dump(vpB, open("./out/vp_B", "wb"))
    pickle.dump(vpBref, open("./out/vp_Bref", "wb"))
    pickle.dump(style_list, open("./out/style_list", "wb"))
    pickle.dump(style_real_list, open("./out/style_real_list", "wb"))

    content_style_swap(model, content_synthesized, style_synthesized, style_real, vp_synthesized, mask_synthesized, bg_synthesized)
    break
