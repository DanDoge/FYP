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

model.netE_content_real.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/content_style/car_df/2020-01-06/car_df/25_net_E_content_real.pth"))
model.netE_content_depth.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/content_style/car_df/2020-01-06/car_df/25_net_E_content_depth.pth"))
model.netG_real.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/content_style/car_df/2020-01-06/car_df/25_net_G_real.pth"))
model.netE_style.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/content_style/car_df/2020-01-06/car_df/25_net_E_style.pth"))

model.netG_real.eval()
model.netE_content_depth.eval()
model.netE_style.eval()


for epoch in range(1):

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - opt.epoch_count)
        model.set_input(data)
        model.real_B = model.real_A2B
        model.mask_B = model.mask_A
        if model.skip():
            continue

        model.backward_GE()

        '''
        torch.save(model.real_A2B, "./out/save_output_" + str(i))
        torch.save(model.rec_A, "./out/save_target_" + str(i))
        '''
        out_np = model.rec_B[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        im = Image.fromarray((out_np * 255).astype(np.uint8))
        im.save("./out/rec/" + str(i) + ".jpg")

        out_np = model.real_B[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        im = Image.fromarray((out_np * 255).astype(np.uint8))
        im.save("./out/real/" + str(i) + ".jpg")

        out_np = model.fake_B[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        im = Image.fromarray((out_np * 255).astype(np.uint8))
        im.save("./out/fake/" + str(i) + ".jpg")

        novelview = model.apply_mask(model.netG_real(model.content_Aref, torch.cat([model.z_style_B, model.vp_Aref], 1)), model.mask_Aref, model.bg_B)
        out_np = novelview[0].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        im = Image.fromarray((out_np * 255).astype(np.uint8))
        im.save("./out/novelview/" + str(i) + ".jpg")

        if i > 10000:
            break

        #break
    break
