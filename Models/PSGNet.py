""" 
__coding__: utf-8
__Author__: liaoxin
__Time__: 2023/9/18 15:27
__File__: PSGNet.py
__remark__: 
__Software__: PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Sequential
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GINConv as GIN_layer
from torch_geometric.nn import GCNConv as GCN_layer
from torch_geometric.nn import GATConv as GAT_layer
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch import nn, einsum
from einops import rearrange

DIST_KERNELS = {
    'exp': {
        'fn': lambda t: torch.exp(-t),
        'mask_value_fn': lambda t: torch.finfo(t.dtype).max
    },
    'softmax': {
        'fn': lambda t: torch.softmax(t, dim=-1),
        'mask_value_fn': lambda t: -torch.finfo(t.dtype).max
    }
}


def exists(val):
    return val is not None


def default(val, d):
    return d if not exists(val) else val


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=78, heads=6, dim_head=13, Lg=0.5, Ld=0.5, La=1, dist_kernel_fn='exp'):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.La = La
        self.Ld = Ld
        self.Lg = Lg

        self.dist_kernel_fn = dist_kernel_fn

    def forward(self, x, mask=None, adjacency_mat=None, distance_mat=None):
        h, La, Ld, Lg, dist_kernel_fn = self.heads, self.La, self.Ld, self.Lg, self.dist_kernel_fn

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h=h, qkv=3).unbind(dim=-2)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(distance_mat):
            distance_mat = rearrange(distance_mat, 'b i j -> b () i j')

        if exists(adjacency_mat):
            adjacency_mat = rearrange(adjacency_mat, 'b i j -> b () i j')

        if exists(mask):
            mask_value = torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * mask[:, None, None, :]

            # mask attention
            dots.masked_fill_(~mask, -mask_value)

            if exists(adjacency_mat):
                adjacency_mat.masked_fill_(~mask, 0.)

        attn = dots.softmax(dim=-1)

        # sum contributions from adjacency and distance tensors
        attn = attn * La

        if exists(adjacency_mat):
            attn = attn + Lg * adjacency_mat

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class VAE(nn.Module):
    def __init__(
            self,
            *,
            dim_in=78,
            model_dim=78,
            dim_out=78,
            depth=1,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp'
    ):
        super().__init__()

        self.embed_to_model = nn.Linear(dim_in, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList([
                Residual(PreNorm(model_dim, Attention(model_dim, heads=heads, Lg=Lg, Ld=Ld, La=La,
                                                      dist_kernel_fn=dist_kernel_fn))),
                Residual(PreNorm(model_dim, FeedForward(model_dim)))
            ])
            self.layers.append(layer)

        self.norm_out = nn.LayerNorm(model_dim)
        self.ff_out = FeedForward(model_dim, dim_out)

    def forward(
            self,
            x,
            mask=None,
            adjacency_mat=None,
            distance_mat=None
    ):
        x = self.embed_to_model(x)

        for (attn, ff) in self.layers:
            x = attn(
                x,
                mask=mask,
                adjacency_mat=adjacency_mat,
                distance_mat=distance_mat
            )
            x = ff(x)

        x = self.norm_out(x)
        x = x.mean(dim=-2)
        x = self.ff_out(x)
        return x


class PSGNet(torch.nn.Module):
    def __init__(self, output_dim=1, num_features_xd=64,
                 ge_features_dim=1000, num_features_xt=25, embed_dim=128,
                 mut_feature_dim=512, meth_feature_dim=256, connect_dim=128, dropout=0.1):
        super(BeTrsDRP_mut, self).__init__()

        self.mat = VAE(
            dim_in=64,
            model_dim=64,
            dim_out=128,
            depth=2,
            heads=6,
            Lg=0.5,
            Ld=0.5,
            La=1,
            dist_kernel_fn='exp')
        self.conv_gcn = GCN_layer(num_features_xd * 2, num_features_xd * 2)
        #self.fc1_drug = Linear(256, 1500)

        net1 = Sequential(Linear(num_features_xd * 2, num_features_xd * 2), ReLU(), Linear(num_features_xd * 2, num_features_xd * 2))
        self.conv_gin1 = GIN_layer(net1)
        self.bn1 = torch.nn.BatchNorm1d(num_features_xd * 2)

        self.fc1_drug = Linear(num_features_xd * 4, 1500)
        self.fc2_drug = Linear(1500, connect_dim)

        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)


        self.EncoderLayer_mut_1 = nn.TransformerEncoderLayer(d_model=mut_feature_dim, nhead=1, dropout=0.5)
        self.conv_mut_1 = nn.TransformerEncoder(self.EncoderLayer_mut_1, 1)
        self.EncoderLayer_mut_2 = nn.TransformerEncoderLayer(d_model=mut_feature_dim, nhead=1, dropout=0.5)
        self.conv_mut_2 = nn.TransformerEncoder(self.EncoderLayer_mut_2, 1)
        self.fc1_mut = Linear(mut_feature_dim, 2944)
        self.fc2_mut = Linear(2944, connect_dim)

        self.fc1_all = Linear(2 * connect_dim, 1024)
        self.fc2_all = Linear(1024, 128)
        #self.fc3_all = Linear(512, 128)
        #self.fc4_all = Linear(256, 256)
        self.out = Linear(connect_dim, output_dim)

        self.relu = ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        drug_poi_data, drug_edg_index, batch = data.x, data.edge_index, data.batch
        mut_data = data.target

        drug_data = torch.unsqueeze(drug_poi_data, 1)
        drug_data = self.mat(drug_data)
        
        drug_data = self.conv_gcn(drug_data, drug_edg_index)
        drug_data = self.relu(drug_data)

        drug_data = F.relu(self.conv_gin1(drug_data, drug_edg_index))
        drug_data = self.bn1(drug_data)

        #drug_data = self.conv_gcn(drug_data, drug_edg_index)
        #drug_data = self.relu(drug_data)

        drug_data = torch.cat([gmp(drug_data, batch), gap(drug_data, batch)], dim=1)
        drug_data = self.relu(self.fc1_drug(drug_data))
        drug_data = self.dropout(drug_data)
        drug_data = self.fc2_drug(drug_data)

        mut_data = mut_data[:, None, :]
        mut_data = self.conv_mut_1(mut_data)
        mut_data = self.conv_mut_2(mut_data)
        mut_data = mut_data.view(-1, mut_data.shape[1] * mut_data.shape[2])
        mut_data = self.fc1_mut(mut_data)
        mut_data = self.dropout(self.relu(mut_data))
        mut_data = self.fc2_mut(mut_data)
        concat_data = torch.cat((drug_data, mut_data), 1)


        concat_data = self.fc1_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)
        concat_data = self.fc2_all(concat_data)
        concat_data = self.relu(concat_data)
        concat_data = self.dropout(concat_data)

        out = self.out(concat_data)
        out = nn.Sigmoid()(out)
        return out, drug_data
