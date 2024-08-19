__all__ = ['DyPTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.Dyconv import Dynamic_conv1d, SELayer2d, SELayer1d
from layers.RevIN import RevIN
from mamba_ssm import Mamba
from layers.MoE3 import MoE
from layers.Mamba_EncDec import Encoder, EncoderLayer
from zeta.nn import FeedForward


# Cell
class DyPTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 entype:str='se', postype:str='w', ltencoder:str='mamba',
                 K:int=6, conv_stride:int=8, conv_kernel_size:int=16,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 dps=[0.3]*5, d_state:int=21, num_x:int=4, topk:int=2, expand:int=2,
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False,
                 device='cpu', **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        if entype == 'dyconv':
            hidden_dim = int((math.ceil((patch_len - conv_kernel_size) / conv_stride) + 1) * d_model)
        elif entype in ['w','se', 'moe']:
            hidden_dim = d_model

        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                entype=entype, postype=postype, ltencoder=ltencoder,
                                n_layers=n_layers, d_model=hidden_dim, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                K=K, conv_stride=conv_stride, conv_kernel_size=conv_kernel_size,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                dps=dps, d_state=d_state, num_x=num_x, topk=topk, expand=expand,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, 
                                device=device, **kwargs)

        # Head
        self.head_nf = hidden_dim * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        elif head_type == 'mix':
            self.head =nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, int(target_window * 2)),
            nn.GELU(),
            nn.Linear(int(target_window * 2), target_window),
            )
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Encoder_MM(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MM, self).__init__()
        self.MA_layers = nn.ModuleList(
            [
                Encoder_mamba(d_model, device) for i in range(2 * n_layers)
            ]
        )
    def forward(self, x):
        for i in range(len(self.MA_layers)):
            x = self.MA_layers[i](x)
        return x


class Encoder_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Encoder_conv, self).__init__()
        self.net = nn.Sequential(Transpose(2,1),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                Transpose(1,2))
        self.net2 = nn.Sequential(Transpose(2,1),
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                Transpose(1,2))
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(out_channels), Transpose(1,2))
        
    def forward(self, x):
        src = x
        x = self.net(x)
        ret = src + self.dropout(x)
        ret = self.norm(ret)
        return ret
    
                
class Encoder_MFCA_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        expand=2,
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MFCA_block, self).__init__()
        self.mamba = Encoder_mamba(d_model=d_model, device=device, d_state=d_state)
        self.ffn = FeedForward(d_model, d_model, expand, dropout=dps[1])
        self.conv = Encoder_conv(d_model, d_model, kernel_size=3, padding='same')
        # self.tcn = Block(3, 3, 128, 128, 7)
        self.dp = nn.Dropout(dps[2])
        self.tf = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=dps[3], dropout=dps[4], n_layers=1,
                                                 activation=activation, res_attention=res_attention,
                                                 pre_norm=pre_norm, store_attn=store_attn)
        self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
    def forward(self, x):
        x = self.mamba(x)
        x = x + self.dp(self.ffn(x))
        x = self.norm1(x)
        # x = x.transpose(-1, -2)
        # x = x.reshape(512, 7, 128, 42)
        # x = self.tcn(x)
        # x = x.transpose(-1, -2)
        # x = x.reshape(512*7, 42, 128)
        x = x + self.dp(self.conv(x))
        x = self.norm1(x)
        x = self.tf(x)
        return x

class Encoder_MFCA(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        expand=2,
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MFCA, self).__init__()
        self.blocks = nn.ModuleList(
            [Encoder_MFCA_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state,
                                   expand=expand,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)]
                                   )

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(x)
        return output

class Encoder_AM(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AM, self).__init__()
        self.n_layers = n_layers
        self.MA_layers = nn.ModuleList(
            [
                # SwitchMoE(dim=d_model, output_dim=d_model, mult=2, num_experts=16),
                TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                             attn_dropout=attn_dropout, dropout=dropout,
                                            activation=activation, res_attention=res_attention,
                                            pre_norm=pre_norm, store_attn=store_attn),

                Encoder_mamba(d_model, device),
            ]
        )
    def forward(self, x):
        for i in range(len(self.MA_layers)):
            x = self.MA_layers[i](x)
        return x


class Encoder_MMA(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MMA, self).__init__()
        self.MA_layers = nn.ModuleList(
            [
               Encoder_MMA_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)
            ]
        )
    def forward(self, x):
        for i in range(len(self.MA_layers)):
            x = self.MA_layers[i](x)
        return x

class Encoder_MMA_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MMA_block, self).__init__()
        self.n_layers = n_layers
        self.mamba1 = Encoder_mamba(d_model, device)
        self.mamba2 = Encoder_mamba(d_model, device)
        self.tf = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                             attn_dropout=attn_dropout, dropout=dropout,
                                            activation=activation, res_attention=res_attention,
                                            pre_norm=pre_norm, store_attn=store_attn)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.mamba1(x)
        x = self.mamba2(x)
        x = self.tf(x)
        return x

class Encoder_AMA(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AMA, self).__init__()
        self.MA_layers = nn.ModuleList(
            [
               Encoder_AMA_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)
            ]
        )
    def forward(self, x):
        for i in range(len(self.MA_layers)):
            x = self.MA_layers[i](x)
        return x

class Encoder_AMA_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AMA_block, self).__init__()
        self.n_layers = n_layers
        self.mamba = Encoder_mamba(d_model, device)
        self.tf1 = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                             attn_dropout=attn_dropout, dropout=dropout,
                                            activation=activation, res_attention=res_attention,
                                            pre_norm=pre_norm, store_attn=store_attn)
        self.tf2 = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                             attn_dropout=attn_dropout, dropout=dropout,
                                            activation=activation, res_attention=res_attention,
                                            pre_norm=pre_norm, store_attn=store_attn)
    def forward(self, x):
        x = self.tf1(x)
        x = self.mamba(x)
        x = self.tf2(x)
        return x

class Encoder_AMM(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AMM, self).__init__()
        self.MA_layers = nn.ModuleList(
            [
               Encoder_AMM_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)
            ]
        )
    def forward(self, x):
        for i in range(len(self.MA_layers)):
            x = self.MA_layers[i](x)
        return x

class Encoder_AMM_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AMM_block, self).__init__()
        self.n_layers = n_layers
        self.mamba1 = Encoder_mamba(d_model, device, d_state=21)
        self.mamba2 = Encoder_mamba(d_model, device, d_state=21)
        self.tf = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                             attn_dropout=attn_dropout, dropout=dropout,
                                            activation=activation, res_attention=res_attention,
                                            pre_norm=pre_norm, store_attn=store_attn)
    def forward(self, x):
        x = self.tf(x)
        x = self.mamba1(x)
        x = self.mamba2(x)
        return x

class Encoder_MAM(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MAM, self).__init__()
        self.MA_layers = nn.ModuleList(
            [
               Encoder_MAM_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)
            ]
        )
    def forward(self, x):
        for i in range(len(self.MA_layers)):
            x = self.MA_layers[i](x)
        return x

class Encoder_MAM_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MAM_block, self).__init__()
        self.n_layers = n_layers
        self.mamba1 = Encoder_mamba(d_model, device, d_state=21)
        self.mamba2 = Encoder_mamba(d_model, device, d_state=21)
        self.tf = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                             attn_dropout=attn_dropout, dropout=dropout,
                                            activation=activation, res_attention=res_attention,
                                            pre_norm=pre_norm, store_attn=store_attn)
    def forward(self, x):
        x = self.mamba1(x)
        x = self.tf(x)
        x = self.mamba2(x)
        return x

class Encoder_MFA(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MFA, self).__init__()
        self.blocks = nn.ModuleList(
            [Encoder_MFA_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)]
                                   )

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(x)
        return output

class Encoder_MFA_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_MFA_block, self).__init__()
        self.mamba = Encoder_mamba(d_model=d_model, device=device, d_state=d_state)
        self.ffn = FeedForward(d_model, d_model, 2, dropout=dps[1])
        self.dp = nn.Dropout(dps[2])
        self.tf = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=dps[3], dropout=dps[4], n_layers=1,
                                                 activation=activation, res_attention=res_attention,
                                                 pre_norm=pre_norm, store_attn=store_attn)
        self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
    def forward(self, x):
        x = self.mamba(x)
        x = x + self.dp(self.ffn(x))
        x = self.norm1(x)
        x = self.tf(x)
        return x


class Encoder_AFM(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AFM, self).__init__()
        self.blocks = nn.ModuleList(
            [Encoder_AFM_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)]
                                   )

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(x)
        return output

class Encoder_AFM_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AFM_block, self).__init__()
        self.mamba = Encoder_mamba(d_model=d_model, device=device, d_state=d_state)
        self.ffn = FeedForward(d_model, d_model, 2, dropout=dps[1])
        self.dp = nn.Dropout(dps[2])
        self.tf = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=dps[3], dropout=dps[4], n_layers=1,
                                                 activation=activation, res_attention=res_attention,
                                                 pre_norm=pre_norm, store_attn=store_attn)
        self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
    def forward(self, x):
        x = self.tf(x)
        x = x + self.dp(self.ffn(x))
        x = self.norm1(x)
        x = self.mamba(x)
        return x

class Encoder_AFCM(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AFCM, self).__init__()
        self.blocks = nn.ModuleList(
            [Encoder_AFCM_block(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state,
                                   pre_norm=pre_norm, activation=activation, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device) for i in range(n_layers)]
                                   )

    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block(x)
        return output

class Encoder_AFCM_block(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm',
                        dps=[0.3, 0.3, 0.3, 0., 0.1],
                        d_state=21,
                        activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, device='cpu'):
        super(Encoder_AFCM_block, self).__init__()
        self.tf = TSTEncoder(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                 attn_dropout=dps[3], dropout=dps[4], n_layers=1,
                                                 activation=activation, res_attention=res_attention,
                                                 pre_norm=pre_norm, store_attn=store_attn)
        self.ffn = FeedForward(d_model, d_model, 2, dropout=dps[1])
        self.conv = Encoder_conv(d_model, d_model, kernel_size=3, padding='same')
        self.dp = nn.Dropout(dps[2])
        self.mamba = Encoder_mamba(d_model=d_model, device=device, d_state=d_state)
        self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
    def forward(self, x):
        x = self.tf(x)
        x = x + self.dp(self.ffn(x))
        x = self.norm1(x)
        x = x + self.dp(self.conv(x))
        x = self.norm2(x)
        x = self.mamba(x)
        return x

class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 entype='se', postype='w', ltencoder='mamba',
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 K=6, conv_stride=8, conv_kernel_size=16,
                 expand=2,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 dps=[0.3]*5, d_state=21, num_x=4, topk=2,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, 
                 device='cpu', **kwargs):
        
        
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.entype = entype
        self.ltencoder = ltencoder
        self.dyconv = Dynamic_conv1d(patch_len=patch_len, out_planes=d_model, kernel_size=conv_kernel_size, ratio=5, K=K, padding=0, stride=conv_stride)
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.se2d = nn.Sequential(
            nn.Conv2d(patch_len, d_model, 1),
            SELayer2d(d_model, reduction=2))
        
        self.se1d = nn.Sequential(
            nn.Conv2d(patch_len, d_model, 1),
            SELayer1d(d_model, reduction=2))

        # self.moe = SwitchMoE(dim=patch_len, output_dim=d_model, mult=4, num_experts=4)
        # self.moe = SwitchMixtureOfExperts(input_dim=patch_len, hidden_dim=d_model * 2, expert_output_dim=d_model, num_experts=2, top_k=1)
        self.moe = MoE(input_size=patch_len, output_size=d_model, num_experts=num_x, hidden_size=d_model, k=topk, noisy_gating=True)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.postype = postype

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        if ltencoder =='tf':
            self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        elif ltencoder =='mamba':
            self.encoder = Encoder_mamba(d_model=d_model, device=device)
        elif ltencoder == 'mma':
            self.encoder = Encoder_MMA(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'amm':
            self.encoder = Encoder_AMM(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'ama':
            self.encoder = Encoder_AMA(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'mam':
            self.encoder = Encoder_MFCA(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state, expand=expand,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'mam_true':
            self.encoder = Encoder_MAM(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'mfa':
            self.encoder = Encoder_MFA(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'afm':
            self.encoder = Encoder_AFM(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'afcm':
            self.encoder = Encoder_AFCM(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dps=dps, d_state=d_state,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'mm':
            self.encoder = Encoder_MM(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        elif ltencoder == 'sm':
            self.encoder = Encoder([EncoderLayer(Mamba(d_model=d_model, d_state=2,d_conv=2,expand=1,),
                        Mamba(d_model=d_model, d_state=2, d_conv=2,expand=1,),d_model, d_model, dropout=dropout, activation=act) for l in range(1)],norm_layer=torch.nn.LayerNorm(d_model)) 
        elif ltencoder == 'am':
            self.encoder = Encoder_AM(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn, device=device)
        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        bs, nvars, patch_len, patch_num = x.shape
        
        # Input encoding
        if self.entype == "w":
            x = x.permute(0,1,3,2)                                                   # x: [bs x n_vars x patch_num x patch_len]
            x = self.W_P(x)                                                          # x: [bs x n_vars x patch_num x patch_len]
        elif self.entype == "dyconv":
            x = x.permute(0,1,3,2)                                                   # x: [bs x n_vars x patch_num x patch_len]
            x = x.reshape(bs * nvars * patch_num, patch_len)
            patch, softmax_att, kernel_wise = self.dyconv(x)
            x = patch.reshape(bs, nvars, patch_num, -1)                              # x: [bs x nvars x patch_num X hidden_dim]
            softmax_att = softmax_att.reshape(bs, nvars, patch_num, -1)              # softmax_att: [bs x nvars x patch_num x K]
            # output 
        elif self.entype == "se":
            x = x.permute(0,2,1,3)                                                   # x: [bs x patch_len x nvars x patch_num]
            x = self.se1d(x)
            x = x.permute(0,2,3,1)                                                   # x: [bs x nvars x patch_num, patch_len]
        elif self.entype == "moe":
            x = x.permute(0,1,3,2)
            x = x.reshape(bs * nvars, patch_num, patch_len)
            x = self.moe(x)
            x = x.reshape(bs, nvars, patch_num, -1)
            
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x hidden_dim]
        
        if self.postype == 'w':
            u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x hidden_dim]
        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x -1]
        # z = self.encoder_gnn(u, softmax_att)
        # z = self.encoder_mamba(u)
        z = torch.reshape(z, (bs, nvars, patch_num, -1))                # z: [bs x nvars x patch_num x -1]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x -1 x patch_num]
        
        return z    
            
# Cell
class Encoder_mamba(nn.Module):
    def __init__(self, d_model, device, m_dp=0.1, d_state:int=4, d_conv:int=4, expand:int=2):
        super().__init__()
        self.mamba = Mamba(
        # This module uses roughly 2 * expand * d_model^2 parameters
        d_model=d_model,# Model dimension d_model
        d_state=d_state,  # SSM state expansion factor, typically 63 or 128
        d_conv=d_conv,    # Local convolution width
        expand=expand,    # Block expansion factor 
        ).to(device)
        self.dropout= nn.Dropout(m_dp)
        self.norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
    
    def forward(self, x):
        z = self.mamba(x)
        z = x + self.dropout(z)            # res
        z = self.norm(z)
        return z
   
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
                

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights