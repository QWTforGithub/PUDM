import os
import open3d as o3d
import numpy as np
import argparse
import torch
from tqdm import tqdm
from einops import rearrange
import open3d
import copy
import os
import glob
import re
from chainer import cuda
from pointnet2.util import normalize_point_cloud
import pickle
import json

def add_possion_noise(pts, sigma, clamp, rate=3.0):
    # input: (b, 3, n)

    assert (clamp > 0)
    poisson_distribution = torch.distributions.Poisson(rate)
    jittered_data = torch.clamp(sigma * poisson_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

def add_laplace_noise(pts, sigma, clamp,loc=0.0,scale=1.0):
    # input: (b, 3, n)

    assert (clamp > 0)
    laplace_distribution = torch.distributions.Laplace(loc=loc, scale=scale)
    jittered_data = torch.clamp(sigma * laplace_distribution.sample(pts.shape), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

def add_gaussian_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

def add_random_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.rand_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data


if __name__ == '__main__':


    mesh_dir="/mnt/SG10T/DataSet/PUGAN/test/mesh"
    save_dir = "/mnt/SG10T/DataSet/PUGAN/temp"
    input_pts_num = 2048
    R=4
    noise_level=0.1
    noise_type="possion" # random, gaussian, laplace, possion


    parser = argparse.ArgumentParser(description='PU-GAN Test Data Generation Arguments')
    parser.add_argument('--input_pts_num', default=input_pts_num, type=int, help='the input points number')
    parser.add_argument('--R', default=R, type=int, help='ground truth for up rate')
    parser.add_argument('--noise_level', default=noise_level, type=float, help='the noise level')
    parser.add_argument('--noise_type', default=noise_type, help='random/gaussian/laplace/possion')
    parser.add_argument('--jitter_max', default=0.03, type=float, help="jitter max")
    parser.add_argument('--mesh_dir', default=mesh_dir, type=str, help='input mesh dir')
    parser.add_argument('--save_dir', default=save_dir, type=str, help='output point cloud dir')
    args = parser.parse_args()

    gt_pts_num = args.input_pts_num * args.R

    print(f"---- points : {input_pts_num}, R : {R}, noise_type : {noise_type}, noise level : {noise_level}----")

    dir_name = 'input_' + str(args.input_pts_num)
    if gt_pts_num % args.input_pts_num == 0:
        up_rate = gt_pts_num / args.input_pts_num
        dir_name += '_' + str(int(up_rate)) + 'X'
    else:
        up_rate = gt_pts_num / args.input_pts_num
        dir_name += '_' + str(up_rate) + 'X'
    if args.noise_level != 0:
        dir_name += f'_{args.noise_type}_' + str(args.noise_level)
    input_save_dir = os.path.join(args.save_dir, dir_name, 'input_' + str(args.input_pts_num))
    if not os.path.exists(input_save_dir):
        os.makedirs(input_save_dir)
    gt_save_dir = os.path.join(args.save_dir, dir_name, 'gt_' + str(gt_pts_num))
    if not os.path.exists(gt_save_dir):
        os.makedirs(gt_save_dir)
    mesh_path = glob.glob(os.path.join(args.mesh_dir, '*.off'))
    for i, path in tqdm(enumerate(mesh_path), desc='Processing'):
        pcd_name = path.split('/')[-1].replace(".off", ".xyz")
        mesh = o3d.io.read_triangle_mesh(path)
        # input pcd
        # input_pcd = mesh.sample_points_poisson_disk(args.input_pts_num)
        input_pcd = mesh.sample_points_poisson_disk(args.input_pts_num)
        input_pts = np.array(input_pcd.points)

        # add noise
        if args.noise_level != 0:
            input_pts = torch.from_numpy(input_pts).float().cuda()
            # (n, 3) -> (3, n)
            input_pts = rearrange(input_pts, 'n c -> c n').contiguous()
            # (3, n) -> (1, 3, n)
            input_pts = input_pts.unsqueeze(0)
            # normalize input
            input_pts, centroid, furthest_distance = normalize_point_cloud(input_pts)
            # add noise
            if(args.noise_type == "gaussian"):
                input_pts = add_gaussian_noise(input_pts, sigma=args.noise_level, clamp=args.jitter_max)
            elif(args.noise_type == "random"):
                input_pts = add_random_noise(input_pts, sigma=args.noise_level, clamp=args.jitter_max)
            elif (args.noise_type == "laplace"):
                input_pts = add_laplace_noise(input_pts, sigma=args.noise_level, clamp=args.jitter_max)
            elif (args.noise_type == "possion"):
                input_pts = add_possion_noise(input_pts, sigma=args.noise_level, clamp=args.jitter_max)
            input_pts = centroid + input_pts * furthest_distance
            # (1, 3, n) -> (n, 3)
            input_pts = rearrange(input_pts.squeeze(0), 'c n -> n c').contiguous()
            input_pts = input_pts.detach().cpu().numpy()

        input_save_path = os.path.join(input_save_dir, pcd_name)
        np.savetxt(input_save_path, input_pts, fmt='%.6f')

        # gt pcd
        gt_pcd = mesh.sample_points_poisson_disk(gt_pts_num)
        gt_pts = np.array(gt_pcd.points)
        gt_save_path = os.path.join(gt_save_dir, pcd_name)
        np.savetxt(gt_save_path, gt_pts, fmt='%.6f')


