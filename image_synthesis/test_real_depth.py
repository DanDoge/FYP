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

model.netG_real.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/depth_style/car_df/2020-03-18/car_df/latest_net_G_real.pth"))
model.netG_depth.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/depth_style/car_df/2020-03-18/car_df/latest_net_G_depth.pth"))
model.netE_style.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/depth_style/car_df/2020-03-18/car_df/latest_net_E_style.pth"))

model.netG_real.eval()
model.netG_depth.eval()
model.netE_style.eval()

vpB = []
vpBref = []

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


for epoch in range(1):

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - opt.epoch_count)
        model.set_input(data)
        if model.skip():
            continue

        #model.backward_GE()
        savefig(model.real_B, "real", i)
        realB_with_vp = cat_feature(model.real_B, model.vp_B)
        model.z_style_B, mu_style_B, logvar_style_B = model.encode_style(realB_with_vp, model.vae)


        model.fake_B = model.apply_mask(model.netG_real(model.real_A, model.vp_A, model.z_style_B), model.mask_A, model.bg_B)
        savefig(model.fake_B, "fake", i)

        model.fake_Brandom = model.apply_mask(model.netG_real(model.real_A, model.vp_A, model.z_texture), model.mask_A, model.bg_B)
        savefig(model.fake_Brandom, "random", i)

        vpB.append(model.vp_B[0].detach().cpu().numpy())
        vpBref.append(model.vp_Bref[0].detach().cpu().numpy())


        #break

    import pickle
    pickle.dump(vpB, open("./out/vp_B", "wb"))
    pickle.dump(vpBref, open("./out/vp_Bref", "wb"))
    break
