import os
import numpy as np
import torch
import pickle
import h5py
import random
import math
from datetime import datetime
from einops import rearrange, repeat
from pointops.functions import pointops
import torch.backends.cudnn as cudnn

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def add_noise(pts, sigma, clamp):
    # input: (b, 3, n)

    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data


def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance


def get_random_seed():
    seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)
    # 1，GT，sample
    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def midpoint_interpolate(sparse_pts, up_rate=4, normal=False, ):
    # sparse_pts: (b, 3, 256)

    if (normal):
        sparse_pts, centroid, furthest_distance = normalize_point_cloud(sparse_pts)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    if (normal):
        interpolated_pts = centroid + interpolated_pts * furthest_distance

    return interpolated_pts


# load and normalize data
def load_h5_data(
        h5_file_path,
        num_points=256,
        R=4
):
    num_out_points = int(num_points * R)
    with h5py.File(h5_file_path, 'r') as f:
        input = f['poisson_%d' % num_points][:]
        gt = f['poisson_%d' % num_out_points][:]

    # (b, n, c)
    assert input.shape[0] == gt.shape[0]

    # the center point of input，中
    input_centroid = np.mean(input, axis=1, keepdims=True)
    input = input - input_centroid
    # (b, 1)，
    input_furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True)
    # normalize to a unit sphere
    input = input / np.expand_dims(input_furthest_distance, axis=-1)
    gt = gt - input_centroid
    gt = gt / np.expand_dims(input_furthest_distance, axis=-1)

    return input, gt


def pc_normalization(input):
    fige = False
    if (isinstance(input, torch.Tensor)):
        input = input.detach().cpu().numpy()
        fige = True
        # the center point of input，
    input_centroid = np.mean(input, axis=1, keepdims=True)
    input = input - input_centroid
    # (b, 1)，
    input_furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True)
    # normalize to a unit sphere
    input = input / np.expand_dims(input_furthest_distance, axis=-1)

    if (fige):
        input = torch.from_numpy(input).cuda()

    return input


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=''):
        self.reset()
        # name is the name of the quantity that we want to record, used as tag in tensorboard
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, summary_writer=None, global_step=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not summary_writer is None:
            # record the val in tensorboard
            summary_writer.add_scalar(self.name, val, global_step=global_step)


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def rescale(x):
    """
    Rescale a tensor to 0-1
    """

    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path, ckpt_name, mode='max', return_num_ckpts=False):
    """
    Find maximum epoch/iteration in path, formatted ${ckpt_name}_${n_iter}.pkl

    Parameters:
    path (str):         checkpoint path
    ckpt_name (str):    name of checkpoint
    mode (str): could be max, all, or best
        for best mode, we find the epoch with the lowest cd loss on test set

    Returns:
    maximum epoch/iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    # epoch = -1
    iterations = []
    for f in files:
        if len(f) <= len(ckpt_name) + 5:
            continue
        if f[:len(ckpt_name)] == ckpt_name and f[-4:] == '.pkl' and ('best' not in f):
            number = f[len(ckpt_name) + 1:-4]
            iterations.append(int(number))
    if return_num_ckpts:
        num_ckpts = len(iterations)
    if len(iterations) == 0:
        if return_num_ckpts:
            return -1, num_ckpts
        return -1
    if mode == 'max':
        if return_num_ckpts:
            return max(iterations), num_ckpts
        return max(iterations)
    elif mode == 'all':
        iterations = sorted(iterations, reverse=True)
        if return_num_ckpts:
            return iterations, num_ckpts
        return iterations
    elif mode == 'best':
        eval_file_name = os.path.join(path, '../../eval_result/gathered_eval_result.pkl')
        handle = open(eval_file_name, 'rb')
        data = pickle.load(handle)
        handle.close()
        cd = np.array(data['avg_cd'])
        idx = np.argmin(cd)
        itera = data['iter'][idx]
        print('We find iteration %d which has the lowest cd loss %.8f' % (itera, cd[idx]))
        if return_num_ckpts:
            return itera, num_ckpts
        return itera
    else:
        raise Exception('%s mode is not supported' % mode)


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print(f"{net.__class__.__name__} Parameters: {params}")
        # print("{} Parameters: {:.6f}M".format(
        #     net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(diffusion_steps.device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                    1 - Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams

def get_rate_list(R=4,base=4):
    ls = []
    l = math.floor(math.log(R,base))
    if(l >= 1): ls = [4] * l
    if(R - np.power(base,l) > 0): ls.append(2)
    return ls

def get_interpolate(point, R=4, base=4):

    ls = get_rate_list(R,base)
    i = point.permute(0, 2, 1)
    for r in ls:
        i = midpoint_interpolate(i, up_rate=r, normal=True)
    return i.permute(0, 2, 1)

def sampling(
        net,
        size,
        diffusion_hyperparams,
        print_every_n_steps=100,
        label=0,
        condition=None,
        R=4,
        gamma=0.5
):
    print("---- DDPM Sampling ----")

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('---- begin sampling, total steps : %s ----' % T)
    z = std_normal(size)
    x = z
    if not label is None and isinstance(label, int):
        label = torch.ones(size[0]).long().cuda() * label
    start_iter = T - 1

    i = get_interpolate(condition,R)
    with torch.no_grad():
        print('reverse step: %d' % T, flush=True)
        for t in range(start_iter, -1, -1): # t from T-1 to 0
            if t % print_every_n_steps == 0:
                print('reverse step: %d' % t, flush=True)
            diffusion_steps = (t * torch.ones((size[0],))).cuda()
            x_ = torch.cat([x, i], dim=-1)
            results = net(x_, condition, ts=diffusion_steps, label=label, use_retained_condition_feature=True)
            if (isinstance(results, tuple)):
                epsilon_theta, condition_pre = results
            else:
                epsilon_theta = results


            # ---- xt -> xt-1 ----
            # xt_1 = 1/sqrt(at)[xt-1 - (1-at)/sqrt(1-at_) * epsilon] ==> (T)
            item_1 = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            # xt_2 = sigma * z ==> (T-1) t > 0
            item_2 = Sigma[t] * std_normal(size) if t > 0 else 0.0
            # xt = gamma * (xt_1 + xt_2 + i) ==> q(xt-1|xt)
            x = gamma * (item_1 + item_2 + i)
            # ---- xt -> xt-1 ----

    if not condition is None:
        net.reset_cond_features()
    return x,condition_pre,z

def sampling_ddim(
        net,
        size,
        diffusion_hyperparams,
        print_every_n_steps=10,
        label=0,
        condition=None,
        R=4,
        gamma=0.5,
        step = 30
):

    print("---- DDIM Sampling ----")

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('---- begin sampling, total steps : %s ----' % step)
    z = std_normal(size)
    x = z
    if not label is None and isinstance(label, int):
        label = torch.ones(size[0]).long().cuda() * label

    ts1 = torch.linspace(T - 1, step // 2 + 1, (step // 2), dtype=torch.int64)
    ts2 = torch.linspace(step // 2, 0, (step // 2), dtype=torch.int64)
    ts = torch.cat([ts1, ts2], dim=0)
    steps = reversed(range(len(ts)))
    i = get_interpolate(condition,R)
    with torch.no_grad():
        for step,t in zip(steps,ts): # t from T-1 to 0
            if (step + 1) % print_every_n_steps == 0 or step == 0:
                print('reverse step: %d' % (step+1 if step>0 else step), flush=True)
            diffusion_steps = (t * torch.ones((size[0],))).cuda()
            x_ = torch.cat([x, i], dim=-1)
            results = net(x_, condition, ts=diffusion_steps, label=label, use_retained_condition_feature=True)
            if (isinstance(results, tuple)):
                epsilon_theta, condition_pre = results
            else:
                epsilon_theta = results

            # ---- xt -> xt-1 ----
            # x0 = (xt - sqrt(1-at_) * noise) 1 / sqrt(at_)
            x0 = (x - torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha_bar[t])
            if(t > 0):
                # sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_)
                c_xt_1_1 = torch.sqrt(Alpha_bar[t - 1]) * x0
                # sqrt(1 - at-1_) * noise
                c_xt_1_2 = torch.sqrt(1 - Alpha_bar[t - 1]) * epsilon_theta
                # xt-1 = gamma * (xt-1 + i)
                x = gamma * (c_xt_1_1 + c_xt_1_2 + i)
            else:
                x = gamma * (x0 + i)
            # ---- xt -> xt-1 ----

    if not condition is None:
        net.reset_cond_features()
    return x,condition_pre,z

def training_loss(
        net,
        loss_fn,
        x0,
        diffusion_hyperparams,
        label=None,
        condition=None,
        alpha=1.0,
        gamma=None
):
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    B, N, D = x0.shape
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # t ~ U[T]
    z = std_normal(x0.shape)
    # xt = sqrt(at_) * X0 + sqrt(1-at_) * z ==> q(xt|x0)
    xt = torch.sqrt(Alpha_bar[diffusion_steps]) * x0 + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
    i = midpoint_interpolate(condition.permute(0, 2, 1)).permute(0, 2, 1)
    xt = torch.cat([xt, i], dim=-1)
    epsilon_theta = net(
        xt,
        condition,
        ts=diffusion_steps.view(B, ),
        label=label
    )

    if (isinstance(epsilon_theta, tuple)):
        noisy, condition_pre = epsilon_theta
        mse_theta = loss_fn(noisy, z)
        mse_psi = loss_fn(condition_pre, condition)
        loss = mse_theta + alpha * mse_psi
    else:
        loss = loss_fn(epsilon_theta, z)

    return loss


def calc_t_emb(ts, t_emb_dim):
    """
    Embed time steps into a higher dimension space
    """
    assert t_emb_dim % 2 == 0

    # input is of shape (B) of integer time steps
    # output is of shape (B, t_emb_dim)
    ts = ts.unsqueeze(1)
    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.to(ts.device)  # shape (half_dim)
    # ts is of shape (B,1)
    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)

    return t_emb


import re


def find_config_file(file_name):
    if 'config' in file_name and '.json' in file_name:
        if os.path.isfile(file_name):
            return file_name
        else:
            print('The config file does not exist. Try to find other config files in the same directory')
            file_path = os.path.split(file_name)[0]
    else:
        if os.path.isdir(file_name):
            file_path = file_name
        else:
            raise Exception('%s does not exist' % file_name)
    # pdb.set_trace()
    files = os.listdir(file_path)
    files = [f for f in files if ('config' in f and '.json' in f)]
    print('We find config files: %s' % files)
    config = files[0]
    number = -1
    for f in files:
        all_numbers = re.findall(r'\d+', f)
        all_numbers = [int(n) for n in all_numbers]
        if len(all_numbers) == 0:
            this_number = -1
        else:
            this_number = max(all_numbers)
        if this_number > number:
            config = f
            number = this_number
    print('We choose the config:', config)
    return os.path.join(file_path, config)


import open3d


def numpy_to_pc(points):
    pc = open3d.geometry.PointCloud()
    points = open3d.utility.Vector3dVector(points)
    pc.points = points
    return pc


import pdb


def ft(x1, x2, a=0.5):
    return x1 * a + x2 * (1 - a)


if __name__ == '__main__':
    # T = 1000
    # B = 32
    # embed_dim = 128
    # diffusion_steps = torch.randint(T, size=(B,1))
    # t1 = calc_diffusion_step_embedding(diffusion_steps, embed_dim)
    # t2 = calc_t_emb(diffusion_steps.view(B), embed_dim)
    file_name = './exp_shapenet/T1000_betaT0.02_shape_generation_noise_reduce_factor_10_corrected_emd_mean_shape/logs/checkpoint'
    config_file = find_config_file(file_name)
    print(config_file)
    pdb.set_trace()
