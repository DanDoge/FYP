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



def savefig(tensor, name, i):
    out_np = tensor[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
    im = Image.fromarray((out_np * 255).astype(np.uint8))
    im.save("./out/" + name + "/" + str(i) + ".png")
    return

def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat

def get_rgb_image(file_rgb):
    img_rgb = Image.open(file_rgb)
    img_mask_rgb = dataset.dataset.transform_mask(img_rgb)
    mask_rgb = img_mask_rgb[3, :, :]
    mask_rgb = mask_rgb.unsqueeze(0)

    rgb_rgb = img_rgb.convert("RGB")
    rgb_rgb = dataset.dataset.transform_rgb(rgb_rgb)
    rgb_rgb = get_normaliztion()(rgb_rgb)

    return rgb_rgb, mask_rgb

def local_tsfm(vp):
    return (vp.float() - torch.Tensor([180., 0.])) / torch.Tensor([180., 90.])

for epoch in range(1):
    import pickle
    with open("./ap_test_dataset", "rb") as f:
        ap_dataset = pickle.load(f)
    root = '/data1/huangdj/PyTorch-CycleGAN-master/test_data'
    for test_point in ap_dataset:
        model, az, el, az_new, el_new, r = test_point
        src_file = os.path.join(root, \
                         model, \
                         model + '_az_' + str(az) + '_el_' + str(el) + "_albedo.png0001.jpg")
        srcreal_B, srcmask_B = get_rgb_image(src_file)
        srcreal_B.to(model.device)
        srcmask_B.to(model.device)

        dst_file = os.path.join(root, \
                         model, \
                         model + '_az_' + str(az_new) + '_el_' + str(el_new) + "_albedo.png0001.jpg")
        dstreal_B, srcmask_B = get_rgb_image(src_file)
        dstreal_B.to(model.device)
        dstmask_B.to(model.device)

        srcvp = torch.Tensor([[az, el]])
        srcvp = local_tsfm(srcvp).to(model.device)

        dstvp = torch.Tensor([[az_new, el_new]])
        dstvp = local_tsfm(dstvp).to(model.device)


        realB_with_vp = cat_feature(srcreal_B, model.vp_B)
        model.z_style_B, mu_style_B, logvar_style_B = model.encode_style(realB_with_vp, model.vae)

        fake_A = model.apply_mask(model.netG_depth(srcreal_B, srcvp, dstvp), dstmask_B, model.bg_A)
        rec_B = model.apply_mask(model.netG_real(fake_A, dstvp, model.z_style_B), dstmask_B, model.bg_B)

        savefig(model.fake_Breal, "ap_dataset", str(model) + '/' + str(az) + '_' + str(el) + '_' +str(r))
    break
