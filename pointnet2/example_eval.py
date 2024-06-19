import os
import open3d
import numpy as np
import torch

from util import rescale, find_max_epoch, print_size, sampling, sampling_ddim, calc_diffusion_hyperparams, AverageMeter,pc_normalization,numpy_to_pc,pc_normalize

import time

def evaluate(
        net,
        example_file,
        diffusion_hyperparams,
        print_every_n_steps=200,
        scale=1,
        R=4,
        gamma=0.5,
        T=1000,
        step=30,
        save_dir = "./test/xys",
        save_xyz = True,            # pre dense point cloud
        save_sp=True,               # pre sparse point cloud
        save_z = False,             # input Gaussian noise
        save_condition = True,     # input sparse point cloud
        normalization=True
):

    times = 0
    save_path = save_dir
    save_xyz = save_xyz
    save_z = save_z
    save_condition =save_condition
    save_sp = save_sp

    label = torch.from_numpy(np.full(shape=(1,), fill_value=R-1, dtype=np.int64)).cuda()
    pc = open3d.io.read_point_cloud(example_file)
    condition = torch.from_numpy(np.asarray(pc.points, dtype=np.float32)).unsqueeze(dim=0).cuda()
    _,N,C = condition.shape

    print(f"**** {N} -----> {N * R}, ===> Upsampling : {R}x, Example File : {example_file} **** ")

    batch = 1
    num_points = R * condition.shape[1]
    net.reset_cond_features()

    start_time = time.time()

    if(step < T):
        generated_data,condition_pre,z = sampling_ddim(
            net=net,
            size=(batch,num_points,3),
            diffusion_hyperparams=diffusion_hyperparams,
            label=label,
            condition=condition,
            R=R,
            gamma=gamma,
            step=step
        )
    else:
        generated_data,condition_pre,z = sampling(
            net=net,
            size=(batch,num_points,3),
            diffusion_hyperparams=diffusion_hyperparams,
            print_every_n_steps=print_every_n_steps,
            label=label,
            condition=condition,
            R=R,
            gamma=gamma
        )

    end_time = time.time() - start_time
    times += end_time

    generated_data = generated_data/scale
    torch.cuda.empty_cache()


    if(save_xyz):
        # ---- save data ----
        generated_np = generated_data[0].detach().cpu().numpy()
        condition_pre_np = condition_pre[0].detach().cpu().numpy()
        z_np = z[0].detach().cpu().numpy()
        condition_np = condition[0].detach().cpu().numpy()
        name = example_file.split("/")[-1].split(".")[0]

        # ---- normalization ----
        if(normalization):
            generated_np = pc_normalize(generated_np)
            condition_pre_np = pc_normalize(condition_pre_np)
            z_np = pc_normalize(z_np)
            condition_np = pc_normalize(condition_np)
        # ---- normalization ----

        # ---- generated ----
        generated_points = generated_np
        generated_pc = numpy_to_pc(generated_points)
        generated_path = os.path.join(save_path,f"{name}.xyz")
        open3d.io.write_point_cloud(filename=generated_path,pointcloud=generated_pc)
        print(f"---- saving : {generated_path} ----")
        # ---- generated ----

        # ---- input condition ----
        if(save_sp):
            condition_pre_points = condition_pre_np
            condition_pre_pc = numpy_to_pc(condition_pre_points)
            condition_pre_path = os.path.join(save_path,f"{name}_sp.xyz")
            open3d.io.write_point_cloud(filename=condition_pre_path,pointcloud=condition_pre_pc)
            print(f"---- saving : {condition_pre_path} ----")
        # ---- input condition ----

        # ---- z ----
        if(save_z):
            z_points = z_np
            z_pc = numpy_to_pc(z_points)
            z_path = os.path.join(save_path,f"{name}_z.xyz")
            open3d.io.write_point_cloud(filename=z_path,pointcloud=z_pc)
            print(f"---- saving : {z_path} ----")
        # ---- z ----

        # ---- pre condition ----
        if(save_condition):
            condition_points = condition_np
            condition_pc = numpy_to_pc(condition_points)
            condition_path = os.path.join(save_path,f"{name}_condition.xyz")
            open3d.io.write_point_cloud(filename=condition_path,pointcloud=condition_pc)
            print(f"---- saving : {condition_path} ----")
        # ---- pre condition ----
        # ---- save data ----

    print(f"Times : {times}")




