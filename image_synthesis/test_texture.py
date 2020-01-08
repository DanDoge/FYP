# import torch.backends.cudnn as cudnn
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
opt.num_threads = 0
opt.serial_batches = True  # no shuffle
opt.batch_size = 1  # force to be 1
dataset = create_dataset(opt)
dataset_size = len(dataset)
print('#training data = %d' % dataset_size)
model = create_model(opt)
model.setup(opt)
model.netG_AB.eval()
total_steps = 0

model.netG_AB.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/texture_real/car_df/2019-11-15/car_df/10_net_G_AB.pth"))
model.netE.module.load_state_dict(torch.load("/data1/huangdj/VON-master/checkpoints/texture_real/car_df/2019-11-15/car_df/10_net_E.pth"))


for epoch in range(1):

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - opt.epoch_count)
        model.set_input(data)
        if model.skip():
            continue
        model.z_encoded, mu1, logvar1 = model.encode(model.real_A2B, model.vae)
        model.fake_B = model.apply_mask(model.netG_AB(model.real_A, mu1), model.mask_A, model.bg_B)

        torch.save(model.fake_B, "./out/save_output_" + str(i))
        torch.save(model.real_A2B, "./out/save_target_" + str(i))

        #break
    break
