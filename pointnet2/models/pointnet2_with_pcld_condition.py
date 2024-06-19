import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, FeatureMapModule
from pointnet2_ops.pointnet2_utils import QueryAndGroup
from torch.utils.data import DataLoader

# from pointnet2.data import Indoor3DSemSeg
# from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
# from pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
from pointnet2.models.pnet import Pnet2Stage
from pointnet2.models.model_utils import get_embedder

import torch.nn.functional as F
import copy
import numpy as np

# from pointnet2.models.Mink.Img_Encoder import ImageEncoder
# from pointnet2.models.Mink.attention_fusion import AttentionFusion

# ---
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        temp = self.to_kv(context)
        k,v = temp.chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# Convolutional Position Encoding
class ConvPosEnc(nn.Module):
    def __init__(self, dim_q, dim_content, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj_q = nn.Conv1d(
            in_channels=dim_q,
            out_channels=dim_q,
            kernel_size=k,
            stride=1,
            padding=k//2,
            groups=dim_q
        )

        self.proj_content = nn.Conv1d(
            in_channels=dim_content,
            out_channels=dim_content,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim_content
        )

    def forward(self,q,content):
        q = q.permute(0,2,1)
        q = self.proj_q(q) + q
        q = q.permute(0,2,1)

        # B,C,H,W = content.shape
        content = content.permute(0, 2, 1)
        content = self.proj_content(content) + content
        content = content.permute(0,2,1)

        return q,content

# main class
class AttentionFusion(nn.Module):
    def __init__(
        self,
        depth,                                  # Self-Attention deep
        dim,                                    # Q dim
        latent_dim = 512,                       # Content dim
        cross_heads = 1,                        # Cross-Attention Head
        latent_heads = 8,                       # Self-Attention Head
        cross_dim_head = 64,                    # Cross-Attention Head dim
        latent_dim_head = 64,                   # Self-Attention Head dim
        weight_tie_layers = False,
        pe=False
    ):
        super().__init__()

        self.pe = pe
        if(pe):
            # position encoding
            self.cpe = ConvPosEnc(
                dim_q=latent_dim,
                dim_content=dim
            )

        # Cross-Attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        #
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        # Self-Attention
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(
        self,
        data,                           # Content data
        mask = None,                    # mask
        queries_encoder = None,         # Q data
    ):
        b, *_, device = *data.shape, data.device
        x = queries_encoder

        # ---- position encoding ----
        if(self.pe):
            x,data = self.cpe(
                q=x,
                content=data,
            )
        # ---- position encoding ----

        # ---- Cross-Attention----
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x
        # ---- Cross-Attention----


        #  ---- Self-Attention ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        #  ---- Self-Attention ----

        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class PointNorm(nn.Module):
    def __init__(self,dim,t=True):
        super().__init__()
        self.t = t
        self.norm = nn.BatchNorm1d(dim)
    def forward(self,x):
        if(self.t):
            x = x.permute(0,2,1)
            return self.norm(x).permute(0,2,1)
        return self.norm(x)

class PointNet2CloudCondition(PointNet2SemSegSSG):

    def _build_model(self):
        self.l_uvw = None
        self.encoder_cond_features = None
        self.decoder_cond_features = None
        self.global_feature = None

        self.attention_setting = self.hparams.get("attention_setting", None)
        self.FeatureMapper_attention_setting = copy.deepcopy(self.attention_setting)
        if self.FeatureMapper_attention_setting is not None:
            self.FeatureMapper_attention_setting['use_attention_module'] = (
                            self.FeatureMapper_attention_setting['add_attention_to_FeatureMapper_module'])

        self.global_attention_setting = self.hparams.get('global_attention_setting', None)

        self.bn = self.hparams.get("bn", True)
        self.scale_factor = 1
        self.record_neighbor_stats = self.hparams["record_neighbor_stats"]
        if self.hparams["include_class_condition"]:
            self.class_emb = nn.Embedding(self.hparams["num_class"], self.hparams["class_condition_dim"])

        in_fea_dim = self.hparams['in_fea_dim']
        partial_in_fea_dim = self.hparams.get('partial_in_fea_dim', in_fea_dim)
        self.attach_position_to_input_feature = self.hparams['attach_position_to_input_feature']
        if self.attach_position_to_input_feature:
            in_fea_dim = in_fea_dim + 3
            partial_in_fea_dim = partial_in_fea_dim + 3

        self.partial_in_fea_dim = partial_in_fea_dim
        self.include_abs_coordinate = self.hparams['include_abs_coordinate']
        self.pooling = self.hparams.get('pooling', 'max')

        self.network_activation = self.hparams.get('activation', 'relu')
        assert self.network_activation in ['relu', 'swish']
        if self.network_activation == 'relu':
            self.network_activation_function = nn.ReLU(True)
        elif self.network_activation == 'swish':
            self.network_activation_function = Swish()

        self.include_local_feature = self.hparams.get('include_local_feature', True)
        self.include_global_feature = self.hparams.get('include_global_feature', False)

        self.global_feature_dim = None
        remove_last_activation = self.hparams.get('global_feature_remove_last_activation', True)
        if self.include_global_feature:
            self.global_feature_dim = self.hparams['pnet_global_feature_architecture'][1][-1]
            self.global_pnet = Pnet2Stage(
                self.hparams['pnet_global_feature_architecture'][0],
                self.hparams['pnet_global_feature_architecture'][1],
                bn=self.bn,
                remove_last_activation=remove_last_activation
            )

        # ---- t_emb ----
        t_dim = self.hparams['t_dim']
        self.fc_t1 = nn.Linear(t_dim, 4*t_dim)
        self.fc_t2 = nn.Linear(4*t_dim, 4*t_dim)
        self.activation = swish
        # ---- t_emb ----

        self.map_type = self.hparams['map_type']

        if self.include_local_feature:
            # build SA module for condition point cloud
            condition_arch = self.hparams['condition_net_architecture']
            npoint_condition = condition_arch['npoint']#[1024, 256, 64, 16]
            radius_condition = condition_arch['radius']#np.array([0.1, 0.2, 0.4, 0.8])
            nsample_condition = condition_arch['nsample']#[32, 32, 32, 32]
            feature_dim_condition = condition_arch['feature_dim']#[32, 32, 64, 64, 128]
            mlp_depth_condition = condition_arch['mlp_depth']#3
            self.SA_modules_condition = self.build_SA_model(
                npoint_condition,
                radius_condition,
                nsample_condition,
                feature_dim_condition,
                mlp_depth_condition,
                partial_in_fea_dim,
                False,
                False,
                neighbor_def=condition_arch['neighbor_definition'],
                activation=self.network_activation,
                bn=self.bn,
                attention_setting=self.attention_setting
            )


            # build feature transfer modules from condition point cloud to the noisy point cloud x_t at encoder
            mapper_arch = self.hparams['feature_mapper_architecture']
            encoder_feature_map_dim = mapper_arch['encoder_feature_map_dim']#[32, 32, 64, 64]


        # ---- Cross-Attention ----
        q = 128
        kv = 512
        self.att_c = AttentionFusion(
            dim=kv,  # the image channels
            depth=0,  # depth of net (self-attention - Processing的数量)
            latent_dim=q,  # the PC channels
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=32,  # number of dimensions per cross attention head
            latent_dim_head=6,  # number of dimensions per latent self attention head
            pe=False
        )
        self.att_noise = AttentionFusion(
            dim=q,  # the image channels
            depth=0,  # depth of net (self-attention - Processing的数量)
            latent_dim=kv,  # the PC channels
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=32,  # number of dimensions per cross attention head
            latent_dim_head=6,  # number of dimensions per latent self attention head
            pe=False
        )
        # ---- Cross-Attention ----


        # build SA module for the noisy point cloud x_t
        arch = self.hparams['architecture']
        npoint = arch['npoint']#[1024, 256, 64, 16]
        radius = arch['radius']#[0.1, 0.2, 0.4, 0.8]
        nsample = arch['nsample']#[32, 32, 32, 32]
        feature_dim = arch['feature_dim']#[32, 64, 128, 256, 512]
        mlp_depth = arch['mlp_depth']#3
        # if first conv, first conv in_fea_dim + encoder_feature_map_dim[0] -> feature_dim[0]
        # if not first conv, mlp[0] = in_fea_dim + encoder_feature_map_dim[0]
        additional_fea_dim = encoder_feature_map_dim if(self.include_local_feature and self.map_type == "map_feature") else None
        self.SA_modules = self.build_SA_model(
            npoint,
            radius,
            nsample,
            feature_dim,
            mlp_depth,
            in_fea_dim+encoder_feature_map_dim[0] if(self.include_local_feature and self.map_type == "map_feature") else in_fea_dim,
            self.hparams['include_t'],
            self.hparams["include_class_condition"],
            include_global_feature=self.include_global_feature,
            global_feature_dim=self.global_feature_dim,
            additional_fea_dim = additional_fea_dim,
            neighbor_def=arch['neighbor_definition'],
            activation=self.network_activation,
            bn=self.bn,
            attention_setting=self.attention_setting,
            global_attention_setting=self.global_attention_setting)

        if self.include_local_feature:
            # build FP module for condition cloud
            include_grouper_condition = condition_arch.get('include_grouper', False)
            use_knn_FP_condition =  condition_arch.get('use_knn_FP', False)
            K_condition = condition_arch.get('K', 3)
            decoder_feature_dim_condition = condition_arch['decoder_feature_dim']#[32, 32, 64, 64, 128]
            decoder_mlp_depth_condition = condition_arch['decoder_mlp_depth']#3
            assert decoder_feature_dim_condition[-1] == feature_dim_condition[-1]
            self.FP_modules_condition = self.build_FP_model(
                decoder_feature_dim_condition,
                decoder_mlp_depth_condition,
                feature_dim_condition,
                partial_in_fea_dim,
                False,
                False,
                use_knn_FP=use_knn_FP_condition,
                K=K_condition,
                include_grouper = include_grouper_condition,
                radius=radius_condition,
                nsample=nsample_condition,
                neighbor_def=condition_arch['neighbor_definition'],
                activation=self.network_activation, bn=self.bn,
                attention_setting=self.attention_setting)

            # build mapper from condition cloud to input cloud at decoder
            decoder_feature_map_dim = mapper_arch['decoder_feature_map_dim']#[32, 32, 64, 64, 128]


        # build FP module for noisy point cloud x_t
        include_grouper = arch.get('include_grouper', False)
        use_knn_FP =  arch.get('use_knn_FP', False)
        K = arch.get('K', 3)
        decoder_feature_dim = arch['decoder_feature_dim']#[128, 128, 256, 256, 512]
        decoder_mlp_depth = arch['decoder_mlp_depth']#3
        assert decoder_feature_dim[-1] == feature_dim[-1]
        additional_fea_dim = decoder_feature_map_dim[1:] if(self.include_local_feature and self.map_type == "map_feature") else None
        self.FP_modules = self.build_FP_model(
            decoder_feature_dim,
            decoder_mlp_depth,
            feature_dim,
            in_fea_dim,
            self.hparams['include_t'],
            self.hparams["include_class_condition"],
            include_global_feature=self.include_global_feature,
            global_feature_dim=self.global_feature_dim,
            additional_fea_dim=additional_fea_dim,
            use_knn_FP=use_knn_FP,
            K=K,
            include_grouper = include_grouper,
            radius=radius,
            nsample=nsample,
            neighbor_def=arch['neighbor_definition'],
            activation=self.network_activation,
            bn=self.bn,
            attention_setting=self.attention_setting,
            global_attention_setting=self.global_attention_setting
        )

        point_upsample_factor = self.hparams.get('point_upsample_factor', 1)
        if point_upsample_factor > 1:
            if self.hparams.get('include_displacement_center_to_final_output', False):
                point_upsample_factor = point_upsample_factor-1
            self.hparams['out_dim'] = int(self.hparams['out_dim'] * (point_upsample_factor+1))

        input_dim = decoder_feature_dim[0]+3
        if(self.include_local_feature and self.map_type == "map_feature"):
            input_dim = input_dim + decoder_feature_map_dim[0]

        self.fc_layer_noise = nn.Sequential(
            nn.Conv1d(input_dim + 3, 128, kernel_size=1, bias=self.hparams["bias"]),
            nn.GroupNorm(32, 128),
            copy.deepcopy(self.network_activation_function),
            nn.Conv1d(128, self.hparams['out_dim'], kernel_size=1),
        )

        self.condition_loss = self.hparams["condition_loss"]
        if (self.include_local_feature and self.hparams["condition_loss"]):
            self.fc_layer_c = nn.Sequential(
                nn.Conv1d(32 + 3, 128, kernel_size=1, bias=self.hparams["bias"]),
                nn.GroupNorm(32, 128),
                copy.deepcopy(self.network_activation_function),
                nn.Conv1d(128, self.hparams['out_dim'], kernel_size=1),
            )

    def reset_cond_features(self):
        self.l_uvw = None
        self.encoder_cond_features = None
        self.decoder_cond_features = None
        self.global_feature = None

    def forward(
            self,
            pointcloud,
            condition,
            ts=None,
            label=None,
            use_retained_condition_feature=False
    ):

        with torch.no_grad():

            xyz_ori = pointcloud[:,:,0:3] / self.scale_factor
            pointcloud = torch.cat([pointcloud, xyz_ori], dim=2)

            uvw_ori = condition[:,:,0:3] / self.scale_factor
            condition = torch.cat([condition, uvw_ori], dim=2)

            xyz, features = self._break_up_pc(pointcloud)
            xyz = xyz / self.scale_factor
            i_pc = pointcloud[:,:,3:6]

            uvw, cond_features = self._break_up_pc(condition)
            uvw = uvw / self.scale_factor

        if (not ts is None) and self.hparams['include_t']:
            t_emb = calc_t_emb(ts, self.hparams['t_dim'])
            t_emb = self.fc_t1(t_emb)
            t_emb = self.activation(t_emb)
            t_emb = self.fc_t2(t_emb)
            t_emb = self.activation(t_emb)
        else:
            t_emb = None

        if (not label is None) and self.hparams['include_class_condition']:
            # label should be 1D tensor of integers of shape (B)
            class_emb = self.class_emb(label) # shape (B, condition_emb_dim)
        else:
            class_emb = None

        if self.include_global_feature:
            condition_emb = self.global_pnet(i_pc.transpose(1,2))
            second_condition_emb = class_emb if self.hparams['include_class_condition'] else None
        else:
            condition_emb = class_emb if self.hparams['include_class_condition'] else None
            second_condition_emb = None


        l_uvw, l_cond_features = [uvw], [cond_features]
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):

            if self.include_local_feature:
                # [B,(2048,1024,256,64,16),3], [B,(3,32,64,64,128),(2048,1024,256,64,16)]
                # ---- Condition Encoder -----
                li_uvw, li_cond_features = self.SA_modules_condition[i](
                    l_uvw[i],
                    l_cond_features[i],
                    t_emb=None,
                    condition_emb=None,
                    subset=True,
                    record_neighbor_stats=self.record_neighbor_stats,
                    pooling=self.pooling
                )

                l_uvw.append(li_uvw)
                l_cond_features.append(li_cond_features)
                # ---- Condition Encoder -----

            # [B,(2048,1024,256,64,16),3], [B,(3,(35,64),(96,128),(192,256),(320,512)),(2048,1024,256,64,16)]
            # ---- Encoder ----
            input_feature = l_features[i]
            li_xyz, li_features = self.SA_modules[i](
                l_xyz[i],
                input_feature,
                t_emb=t_emb,
                condition_emb=condition_emb,
                second_condition_emb=second_condition_emb,
                subset=True,
                record_neighbor_stats=self.record_neighbor_stats,
                pooling=self.pooling
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            # ---- Encoder ----

        # ---- Cross-Attention ----
        l_cond_features[-1] = self.att_c(
            l_features[-1].permute(0, 2, 1),
            queries_encoder=l_cond_features[-1].permute(0, 2, 1)
        ).permute(0, 2, 1).contiguous()

        l_features[-1] = self.att_noise(
            l_cond_features[-1].permute(0, 2, 1),
            queries_encoder=l_features[-1].permute(0, 2, 1)
        ).permute(0, 2, 1).contiguous()
        # ---- Cross-Attention ----

        if self.include_local_feature:
            if use_retained_condition_feature and self.l_uvw is None:
                self.l_uvw = l_uvw
                self.encoder_cond_features = copy.deepcopy(l_cond_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):

            if self.include_local_feature:
                # [B,(16,64,256,1024,2048),3], [B,(128,64,64,32,32),(16,64,256,1024,2048)]
                # ---- Condition Decoder ----
                l_cond_features[i - 1] = self.FP_modules_condition[i](
                    l_uvw[i - 1],
                    l_uvw[i],
                    l_cond_features[i - 1],
                    l_cond_features[i],
                    t_emb = None,
                    condition_emb=None,
                    record_neighbor_stats=self.record_neighbor_stats,
                    pooling=self.pooling
                )
                # ---- Condition Decoder ----

            # [B,(16,64,256),3], [B,((640,256),(320,256),())]
            # ---- Decoder ----
            input_feature = l_features[i]
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1],
                l_xyz[i],
                l_features[i - 1],
                input_feature,
                t_emb = t_emb,
                condition_emb=condition_emb,
                second_condition_emb=second_condition_emb,
                record_neighbor_stats=self.record_neighbor_stats,
                pooling=self.pooling
            )
            # ---- Decoder ----

        out_feature = l_features[0]
        out_feature = torch.cat([out_feature.transpose(1,2), i_pc, xyz], dim=-1).permute(0,2,1)
        out = self.fc_layer_noise(out_feature).permute(0,2,1)
        if (self.train): out = self.hparams['gamma'] * (out + i_pc)
        if(self.include_local_feature and self.condition_loss):
            condition_feature = l_cond_features[0]
            condition_feature = torch.cat([condition_feature.transpose(1,2), uvw], dim=-1).permute(0,2,1)
            condition_out = self.fc_layer_c(condition_feature).permute(0,2,1)
            return out,condition_out
        else:
            return out


