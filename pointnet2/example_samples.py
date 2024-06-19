import os
import argparse
import json
import torch
from shutil import copyfile

from util import rescale, find_max_epoch, sampling, calc_diffusion_hyperparams, AverageMeter, set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
from example_eval import evaluate
import sys

sys.path.append(os.path.abspath("../"))

def main(
        config_file,
        pointnet_config,
        datasetset_config,
        diffusion_config,
        diffusion_hyperparams,
        batch_size,
        checkpoint_path=None,
        save_dir='',
        example_file='',
        gamma=0.5,
        R=4,
        step=30,
        normalization=True
):
    if (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    print(f"Saving Path: {save_dir}")

    try:
        copyfile(config_file, os.path.join(save_dir, os.path.split(config_file)[1]))
    except:
        print('The two files are the same, donot need to copy')

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    net = PointNet2CloudCondition(pointnet_config).cuda()

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"---- Loading : {checkpoint_path} ----")
    except:
        raise Exception('Model is not loaded successfully')

    # get data loader
    datasetset_config['batch_size'] = batch_size * torch.cuda.device_count()
    datasetset_config['eval_batch_size'] = batch_size * torch.cuda.device_count()
    data_scale = datasetset_config['scale']

    evaluate(
        net=net,
        diffusion_hyperparams=diffusion_hyperparams,
        print_every_n_steps=diffusion_config["T"] // 5,
        scale=data_scale,
        R=R,
        T=diffusion_config["T"],
        step=step,
        save_dir=save_dir,
        gamma=gamma,
        example_file=example_file,
        normalization=normalization
    )


if __name__ == "__main__":

    set_seed(42)
    # using pretrained checkpoint on PUGAN
    dataset = "PUGAN"
    save_dir = f"./test/example"

    example_file = f"./example/KITTI.xyz"
    # example_file = f"./example/ScanNet.xyz"
    # example_file = f"./example/pig_Gaussian_noise_0.01.xyz"
    # example_file = f"./example/pig.xyz"

    R = 4
    device_ids = "1"
    batch_size = 1
    gamma = 0.5
    step = 30
    normalization = True

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('--R', type=int, default=R, help='up rate')
    parser.add_argument('-b', '--batch_size', type=int, default=batch_size, help='batchsize to generate data')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='the directory to save the generated samples')
    parser.add_argument('--example_file', type=str, default=example_file)
    parser.add_argument('--device_ids', type=str, default=device_ids, help='gpu device indices to use')
    parser.add_argument('--gamma', type=float, default=gamma)
    parser.add_argument('--step', type=int, default=step)
    parser.add_argument('--normalization', type=bool, default=normalization)

    args = parser.parse_args()

    args.config = f"./exp_configs/{dataset}.json"
    args.checkpoint_path = f"./pkls/{dataset.lower()}.pkl"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    gen_config = config["gen_config"]
    pointnet_config = config["pointnet_config"]
    diffusion_config = config["diffusion_config"]
    train_config = config["train_config"]

    if train_config['dataset'] == 'PU1K':
        datasetset_config = config["pu1k_dataset_config"]
    elif train_config['dataset'] == 'PUGAN':
        datasetset_config = config["pugan_dataset_config"]
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    with torch.no_grad():
        main(
            config_file=args.config,
            pointnet_config=pointnet_config,
            datasetset_config=datasetset_config,
            diffusion_config=diffusion_config,
            diffusion_hyperparams=diffusion_hyperparams,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint_path,
            save_dir=args.save_dir,
            example_file=args.example_file,
            gamma=args.gamma,
            R=args.R,
            step=args.step,
            normalization=normalization
        )

