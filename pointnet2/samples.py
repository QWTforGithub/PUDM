
import argparse
import json
import torch
from shutil import copyfile

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from util import rescale, find_max_epoch, sampling, calc_diffusion_hyperparams, AverageMeter, set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from dataset import get_dataloader
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict
from eval import evaluate

def main(
        config_file,
        pointnet_config,
        datasetset_config,
        diffusion_config,
        diffusion_hyperparams,
        batch_size,
        phase,
        checkpoint_path=None,
        save_dir='',
        gamma=0.5,
        R=4,
        step=30
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
    testloader = get_dataloader(datasetset_config, phase=phase)
    data_scale = datasetset_config['scale']
    npoints = datasetset_config['npoints']
    compute_cd = True
    data_dir = datasetset_config["data_dir"]

    CD_loss, HD_loss, P2F_loss, total_meta, metrics = evaluate(
        net=net,
        testloader=testloader,
        diffusion_hyperparams=diffusion_hyperparams,
        print_every_n_steps=diffusion_config["T"] // 5,
        scale=data_scale,
        compute_cd=compute_cd,
        return_all_metrics=True,
        R=R,
        npoints=npoints,
        T=diffusion_config["T"],
        step=step,
        save_dir=save_dir,
        gamma=gamma,
        mesh_path=f"{data_dir}/{phase}/mesh"
    )

    print("{} X :: Results: \t{}->{} \tCD loss: {} \tHD loss: {} \tP2F loss: {}".format(
        R,
        npoints,
        npoints * R,
        CD_loss,
        HD_loss,
        P2F_loss
    ), flush=True)

    return CD_loss, HD_loss, P2F_loss


if __name__ == "__main__":

    set_seed(42)
    dataset = "PUGAN"

    device_ids = "1"
    phase = "test"
    R = 4
    batch_size = 14  # 14 : PUGAN, 43 : PU1K
    gamma = 0.5
    step = 30

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('--R', type=int, default=R, help='up rate')
    parser.add_argument('-b', '--batch_size', type=int, default=batch_size, help='batchsize to generate data')
    parser.add_argument('-p', '--phase', type=str, default=phase, help='which part of the dataset to generated samples')
    parser.add_argument('--device_ids', type=str, default=device_ids, help='gpu device indices to use')
    parser.add_argument('--gamma', type=float, default=gamma)
    parser.add_argument('--step', type=int, default=step)
    args = parser.parse_args()

    args.config = f"./exp_configs/{args.dataset}.json"
    args.checkpoint_path = f"./pkls/{args.dataset.lower()}.pkl"
    args.save_dir = f"./test/{args.dataset.lower()}"

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
        CD_loss, HD_loss, P2F_loss = main(
            config_file=args.config,
            pointnet_config=pointnet_config,
            datasetset_config=datasetset_config,
            diffusion_config=diffusion_config,
            diffusion_hyperparams=diffusion_hyperparams,
            batch_size=args.batch_size,
            phase=args.phase,
            checkpoint_path=args.checkpoint_path,
            save_dir=args.save_dir,
            gamma=args.gamma,
            R=args.R,
            step=args.step
        )

