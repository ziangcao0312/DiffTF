from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from timm.models.vision_transformer import  Attention, Mlp

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        #ipdb.set_trace()

        B,C,W,H=x.shape
        x=x.reshape(B,C,W,3,H//3)
        output=[]
        for i in range(3):
            #inputs=x[:,:,:,i,:]
            h = self.in_layers(x[:,:,:,i,:])
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
            output.append(self.skip_connection(x[:,:,:,i,:]) + h)
        
        x=th.stack(output,dim=3).reshape(B,-1,W,H)

        return x

class Attention_qBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))


    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
class Attention_q(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,embchannel=1152):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(dim, inner_dim , bias = False)
        self.key = nn.Linear(dim, inner_dim , bias = False)
        self.value = nn.Linear(dim, inner_dim , bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) 

        # self.emb=nn.Sequential(
        #     nn.Linear(embchannel, inner_dim),
            
        # ) 

        # nn.init.constant_(self.to_out[0].weight, 0)
        # nn.init.constant_(self.to_out[0].bias, 0)


    def forward(self, query,x):
        #ipdb.set_trace()
        b,_,n,d=x.shape
        query=self.query(query)
        key=self.key(x.reshape(b,-1,d))
        value=self.value(x.reshape(b,-1,d))
        
        key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [key,value])  #1 16 65 64

        ff=lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads)

        query=ff(query) 

        #ipdb.set_trace()

        dots = th.matmul(query, key.transpose(-1, -2)) * self.scale  #1 16 65 65

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = th.matmul(attn, value)   #1 16 65 64
        out = rearrange(out, 'b h n d -> b n (h d)') #1  65 1024
        return self.to_out(out)  #1  65 1024
        
def modulatenorm(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)    
class DiTBlock(TimestepBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, imagesize,hidden_size,att_hidden, num_heads, patchsize,mlp_ratio=4.0,time_embed_dim=1024, dropout=0,**block_kwargs):
        super().__init__()
        #patchsize=imagesize//8
        #hidden_size=hidden_size*patchsize*patchsize
        patchnum=(imagesize//patchsize)**2

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 12* hidden_size, bias=True)
        )

        self.att_q1= Attention_q(hidden_size, heads = num_heads, dim_head = att_hidden//num_heads, dropout = dropout)
        # self.att_q2= Attention_q(time_embed_dim, heads = num_heads, dim_head = hidden_size//num_heads, dropout = dropout,embchannel=time_embed_dim)
        # self.att_q3= Attention_q(time_embed_dim, heads = num_heads, dim_head = hidden_size//num_heads, dropout = dropout,embchannel=time_embed_dim)
        # self.weightpatchx=nn.Parameter(th.zeros(1))
        # self.weightpatchy=nn.Parameter(th.zeros(1))
        # self.weightpatchz=nn.Parameter(th.zeros(1))
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        

        # self.mod=nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(time_embed_dim,  time_embed_dim, bias=True)
        # )

        # nn.init.constant_(self.mod[-1].weight, 0)
        # nn.init.constant_(self.mod[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)




    def forward(self, x, emb):
        #ipdb.set_trace()
        
        # attemb=emb.unsqueeze(1).unsqueeze(1)*(1+self.mod(self.pos_embedding))
        # x=x+attemb



        shift_msa, scale_msa,shift_msa1, scale_msa1,shift_msa2, scale_msa2, gate_msa,gate_msa1,gate_msa2, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb).chunk(12, dim=1)
        #ipdb.set_trace()
        depx=modulatenorm(x[:,0,:,:], shift_msa, scale_msa)
        depy=modulatenorm(x[:,1,:,:], shift_msa1, scale_msa1)
        depz=modulatenorm(x[:,2,:,:], shift_msa2, scale_msa2)
        #ipdb.set_trace()
        

        depx = depx + gate_msa.unsqueeze(1) * self.att_q1(depx,th.stack([depx,depy,depz],1))
        depy = depy + gate_msa1.unsqueeze(1) * self.att_q1(depy,th.stack([depx,depy,depz],1))
        depz = depz + gate_msa2.unsqueeze(1) * self.att_q1(depz,th.stack([depx,depy,depz],1))


        # depx = self.att_q1(triplane[:,0,:,:],triplane)+triplane[:,0,:,:]
        # depy = self.att_q2(triplane[:,1,:,:],triplane)+triplane[:,1,:,:]
        # depz = self.att_q3(triplane[:,2,:,:],triplane)+triplane[:,2,:,:]

        x=th.stack([depx,depy,depz],1)

        x = x + gate_mlp.unsqueeze(1).unsqueeze(1) * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x

class Patchemb(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, imagesize,input_channel,hidden_size, num_heads, patchsize,mlp_ratio=4.0,time_embed_dim=1024, dropout=0,**block_kwargs):
        super().__init__()
        #patchsize=imagesize//8
        patchnum=(imagesize//patchsize)**2

        self.setpx=nn.Sequential(
            normalization(input_channel),
            SiLU(),
            conv_nd(2, input_channel, hidden_size, kernel_size=patchsize, stride=patchsize, bias=True),
        )
        
        self.setpy=nn.Sequential(
            normalization(input_channel),
            SiLU(),
            conv_nd(2, input_channel, hidden_size, kernel_size=patchsize, stride=patchsize, bias=True),
        )
        self.setpz=nn.Sequential(
            normalization(input_channel),
            SiLU(),
            conv_nd(2, input_channel, hidden_size, kernel_size=patchsize, stride=patchsize, bias=True),
        )
        self.pos_embedding=nn.Parameter(th.zeros(1, 3,patchnum, hidden_size))
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        
        


    def forward(self, x):
        #ipdb.set_trace()
        b,c,w,hc=x.shape
        x=x.reshape(b,c,w,3,hc//3).permute(0,3,1,2,4)
        #ipdb.set_trace()
        #ipdb.set_trace()
        x=th.stack([self.setpx(x[:,0,:,:,:]).flatten(2).transpose(1, 2),\
                       self.setpy(x[:,1,:,:,:]).flatten(2).transpose(1, 2),\
                        self.setpz(x[:,2,:,:,:]).flatten(2).transpose(1, 2)],1)
        x += self.pos_embedding.repeat(x.shape[0],1,1,1)
        x = self.norm(self.dropout(x))
        
        #ipdb.set_trace()
        


        return x


class Integrate(TimestepBlock):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, imagesize,hidden_size, output_channel,num_heads,patchsize, mlp_ratio=4.0,time_embed_dim=1024, dropout=0,**block_kwargs):
        super().__init__()
        #patchsize=imagesize//8
        ouput_size=output_channel*patchsize*patchsize
        patchnum=(imagesize//patchsize)**2

        self.norm_final1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_final2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_final3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear1 = nn.Linear(hidden_size, ouput_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, ouput_size, bias=True)
        self.linear3 = nn.Linear(hidden_size, ouput_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 6 * hidden_size, bias=True)
        )


        self.imagesize=imagesize
        self.patchsize=patchsize
        self.hidden=ouput_size//(patchsize**2)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)



    def forward(self, x,c):
        #ipdb.set_trace()
        shift1, scale1,shift2, scale2,shift3, scale3 = self.adaLN_modulation(c).chunk(6, dim=1)
        
        depx = modulatenorm(self.norm_final1(x[:,0,:,:]), shift1, scale1)
        depy = modulatenorm(self.norm_final2(x[:,1,:,:]), shift2, scale2)
        depz = modulatenorm(self.norm_final3(x[:,2,:,:]), shift3, scale3)

        x = th.stack([self.linear1(depx),self.linear2(depy),self.linear3(depz)],1)
        #ipdb.set_trace()
        x = x.reshape(shape=(x.shape[0], 3, self.imagesize//self.patchsize, self.imagesize//self.patchsize, self.patchsize, self.patchsize , self.hidden))
        x = th.einsum('nahwpqc->nachpwq', x)
        x = x.reshape(shape=(x.shape[0],3* self.hidden, self.imagesize, self.imagesize))
        b,c,w,h=x.shape
        x=x.reshape(b,3,c//3,w,h).permute(0,2,3,1,4).reshape(b,c//3,w,3*h)
        #ipdb.set_trace()
        #x=self.setp2(x).reshape(x.shape[0],c,self.imagesize,self.imagesize).contiguous() 
        #func=Rearrange('b a (h w) (p1 p2 c)  -> b a c (h p1) (w p2)', p1 = 2, p2 = 2,w=16,h=16)


        return x
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = th.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = th.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class Attention3d(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(dim, inner_dim , bias = False)
        self.key = nn.Linear(dim, inner_dim , bias = False)
        self.value = nn.Linear(dim, inner_dim , bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #ipdb.set_trace()
        b,_,n,d=x.shape
        query=self.query(x.view(b,-1,d))
        key=self.key(x.view(b,-1,d))
        value=self.value(x.view(b,-1,d))


        
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [query,key,value])  #1 16 65 64

        #ipdb.set_trace()

        dots = th.matmul(query, key.transpose(-1, -2)) * self.scale  #1 16 65 65

        attn = self.attend(dots)
        attn = self.dropout(attn)
        #ipdb.set_trace()

        out = th.matmul(attn, value)   #1 16 65 64
        out = rearrange(out, 'b h n d -> b n (h d)') #1  65 1024
        return self.to_out(out).view(b,3,-1,d)  #1  65 1024
    
class Transformer_depend(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,embchannel=1152):
        super().__init__()
        self.encoder = nn.ModuleList([])
        for _ in range(depth):
            self.encoder.append(nn.ModuleList([
                Attention3d(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, nn.Identity()),
                FeedForward(dim, mlp_dim, dropout = dropout),
                PreNorm(dim, nn.Identity()),
            ]))
        #ipdb.set_trace()
        self.decoder = nn.ModuleList([])
        for _ in range(depth):

            self.decoder.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, nn.Identity()),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, nn.Identity()),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, nn.Identity()),

                Attention_q(dim, heads = heads, dim_head = dim_head, dropout = dropout,embchannel=embchannel),
                #PreNorm(dim, nn.Identity()),
                PreNorm(dim, nn.Identity()),
                FeedForward(dim, mlp_dim, dropout = dropout),
                PreNorm(dim, nn.Identity()),
                FeedForward(dim, mlp_dim, dropout = dropout),
                PreNorm(dim, nn.Identity()),
                FeedForward(dim, mlp_dim, dropout = dropout),
                PreNorm(dim, nn.Identity()),
            ]))
    def forward(self, x,emb):
        #ipdb.set_trace()
        for attn, norm1,ff,norm2 in self.encoder:
            x = norm1(attn(x) + x)
            x = norm2(ff(x) + x)
        queryx=x[:,0,:,:]
        queryy=x[:,1,:,:]
        queryz=x[:,2,:,:]
  
        for att_q1,norm1,att_q2,norm2,att_q3,norm3,attn,norm4, ff1,normf1,ff2,normf2,ff3,normf3 in self.decoder:
            
            queryx=norm1(att_q1(queryx)+queryx)
            queryy=norm2(att_q2(queryy)+queryy)
            queryz=norm3(att_q3(queryz)+queryz)
            #ipdb.set_trace()
            queryx = norm4(attn(queryx,x) + queryx)
            queryy = norm4(attn(queryy,x) + queryy)
            queryz = norm4(attn(queryz,x) + queryz)
            #ipdb.set_trace()
            queryx=normf1(ff1(queryx)+queryx)
            queryy=normf2(ff2(queryy)+queryy)
            queryz=normf3(ff3(queryz)+queryz)
        return th.stack([queryx,queryy,queryz],1)
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class Transtri(TimestepBlock):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', in_channels = 32, dim_head = 64, dropout = 0., emb_dropout = 0.,embchannel=1152):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        #ipdb.set_trace()
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding1 = nn.Sequential(
            normalization(in_channels),
            SiLU(),
            conv_nd(2, in_channels, dim, kernel_size=patch_height, stride=patch_height, bias=True),
        )
        
        self.to_patch_embedding2 = nn.Sequential(
            normalization(in_channels),
            SiLU(),
            conv_nd(2, in_channels, dim, kernel_size=patch_height, stride=patch_height, bias=True),
        )
        self.to_patch_embedding3 = nn.Sequential(
            normalization(in_channels),
            SiLU(),
            conv_nd(2, in_channels, dim, kernel_size=patch_height, stride=patch_height, bias=True),
        )
        # nn.Sequential(
        #     Rearrange('b a c (h p1) (w p2) -> b a (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        #     SiLU(),
        #     nn.Linear(dim,dim),
        # )
        self.dropout = nn.Dropout(dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embchannel, 6* dim, bias=True)
        )



        self.transformerx = Transformer_depend(dim, depth, heads, dim_head, mlp_dim, dropout,embchannel)
        # self.transformery = Transformer_depend(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.transformerz = Transformer_depend(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pos_embedding=nn.Parameter(th.zeros(1, 3,num_patches, dim))
        self.pool = pool
        self.to_patch_embeddingx = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_dim, patch_dim),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, w=(image_width // patch_width),h=(image_height // patch_height)),
        )
        self.to_patch_embeddingy = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_dim, patch_dim),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, w=(image_width // patch_width),h=(image_height // patch_height)),
        )
        self.to_patch_embeddingz = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_dim, patch_dim),
            nn.Dropout(dropout),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, w=(image_width // patch_width),h=(image_height // patch_height)),
        )
        # self.mod=nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(dim,  dim, bias=True)
        # )
        self.wegithx=nn.Parameter(th.zeros(1, in_channels,1, 1))
        self.wegithy=nn.Parameter(th.zeros(1, in_channels,1, 1))
        self.wegithz=nn.Parameter(th.zeros(1, in_channels,1, 1))

        # nn.init.constant_(self.mod[-1].weight, 0)
        # nn.init.constant_(self.mod[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


    def forward(self, img,emb):
        #ipdb.set_trace()
        b, c, w,hs = img.shape
        img=img.reshape(b,c,w,3,hs//3).permute(0,3,1,2,4)
        #ipdb.set_trace()
        x = th.stack([self.to_patch_embedding1(img[:,0,:,:,:]).flatten(2).transpose(1, 2),\
                         self.to_patch_embedding2(img[:,1,:,:,:]).flatten(2).transpose(1, 2),\
                            self.to_patch_embedding3(img[:,2,:,:,:]).flatten(2).transpose(1, 2)],1)
        
        weight1,weight2,weight3,bias1,bias2,bias3=self.adaLN_modulation(emb).chunk(6, dim=1)
        #ipdb.set_trace()
        #b, _, n,_ = x.shape

        #cls_tokens = repeat(self.cls_token, '1 1 1 d -> b 3 1 d', b = b)
        #ipdb.set_trace()
        #x = torch.cat((cls_tokens, x), dim=-2)
        #ipdb.set_trace()
        x += self.pos_embedding.repeat(b,1,1,1)
        x = self.dropout(x)
        # attemb=emb.unsqueeze(1).unsqueeze(1)*(1+self.mod(self.pos_embedding))
        # x=x+attemb
        
        #ipdb.set_trace()
        x=self.transformerx(x,emb)
        depx = self.wegithx*self.to_patch_embeddingx((1+weight1.unsqueeze(1))*x[:,0,:,:]+bias1.unsqueeze(1)) +img[:,0,:,:,:]
        depy =  self.wegithy*self.to_patch_embeddingy((1+weight2.unsqueeze(1))*x[:,1,:,:]+bias2.unsqueeze(1)) +  img[:,1,:,:,:]
        depz =  self.wegithz*self.to_patch_embeddingz((1+weight3.unsqueeze(1))*x[:,2,:,:]+bias3.unsqueeze(1))  + img[:,2,:,:,:]
        out=th.cat([depx,depy,depz],1)



        #ipdb.set_trace()

        return out.reshape(b,3,c,w,hs//3).permute(0,2,3,1,4).reshape(b,c,w,hs)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 6
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        self.input_block=TimestepEmbedSequential(
                    conv_nd(dims, in_channels//3, model_channels, 3, padding=1)
                )
        self.input_blocks = nn.ModuleList([])
        
        #patchsize=16
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    # layers.append(
                    #     AttentionBlock(
                    #         ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                    #     )
                    # )
                    patch_size=256//ds//8
                    #if 256//ds==16:
                    input_ch=ch
                    output_ch=ch
                    hiden=ch
                    attn_hid=ch*2

                    layers.append(
                        Patchemb(256//ds, input_ch, hiden, num_heads, patch_size,mlp_ratio=1,time_embed_dim=time_embed_dim),

                    )
                    layers.append(
                        DiTBlock(256//ds, hiden, attn_hid,num_heads,patch_size,mlp_ratio=1,dropout=dropout,time_embed_dim=time_embed_dim),

                    )
                    layers.append(
                        Integrate(256//ds,hiden,output_ch, num_heads, patch_size,mlp_ratio=1,time_embed_dim=time_embed_dim),

                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2
        #ipdb.set_trace()

        self.middle_block = TimestepEmbedSequential(
            Transtri(
                        image_size = 256//ds,
                        patch_size = 2,
                        in_channels=ch,
                        dim = 2048,
                        depth = 4,
                        heads = num_heads,
                        dim_head = 2048//num_heads,
                        mlp_dim = 2048,
                        dropout = dropout,
                        emb_dropout = 0.0,
                        embchannel=time_embed_dim
                        )
        )


        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    patch_size=256//ds//8
                    #if 256//ds==16:
                    input_ch=ch
                    output_ch=ch
                    hiden=ch
                    attn_hid=ch*2

                    layers.append(
                        Patchemb(256//ds, input_ch, hiden, num_heads, patch_size,mlp_ratio=1,time_embed_dim=time_embed_dim),

                    )
                    layers.append(
                        DiTBlock(256//ds, hiden, attn_hid,num_heads,patch_size,mlp_ratio=1,dropout=dropout,time_embed_dim=time_embed_dim),

                    )
                    layers.append(
                        Integrate(256//ds,hiden,output_ch, num_heads, patch_size,mlp_ratio=1,time_embed_dim=time_embed_dim),

                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels//3, 3, padding=1)),
        )
        # self.cnnrefinement=nn.Sequential(
        #     nn.Conv2d(18, 64, kernel_size=3, bias=True, stride=1,padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=3, bias=True, stride=1,padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 64, kernel_size=3, bias=True, stride=1,padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 18, kernel_size=3, bias=True, stride=1,padding=1),
        # )
        # for block in self.middle_block:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        B,C,W,H=x.shape
        x=x.reshape(B,3,C//3,W,H).permute(0,2,3,1,4)
        
        #ipdb.set_trace()
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        output=[]
        for i in range(3):
            output.append(self.input_block(h[:,:,:,i,:],emb))
        h=th.stack(output,dim=3).reshape(B,-1,W,3*H)
        hs.append(h)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        # B,C,W,H=h.shape
        # h=h.reshape(B,C,W,3,H//3).permute(0,3,1,2,4).reshape(B,3*C,W,H)
        #ipdb.set_trace()
        #h=h+self.w*self.transtri1(h.view(B,3,-1,W,H)).view(B,C,W,H)
        h = self.middle_block(h, emb)
        #h=h+self.norm2(self.transtri2(h.view(B,3,-1,W,H)).view(B,C,W,H))
        #ipdb.set_trace()
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        b,c,w1,h1=h.shape
        h=h.reshape(b,c,w1,3,h1//3)
        output=[]
        for i in range(3):
            output.append(self.out(h[:,:,:,i,:]))
        h=th.stack(output,dim=3).permute(0,3,1,2,4).reshape(B,C,W,H)

        #ipdb.set_trace()
        return h

    # def get_feature_vectors(self, x, timesteps, y=None):
    #     """
    #     Apply the model and return all of the intermediate tensors.

    #     :param x: an [N x C x ...] Tensor of inputs.
    #     :param timesteps: a 1-D batch of timesteps.
    #     :param y: an [N] Tensor of labels, if class-conditional.
    #     :return: a dict with the following keys:
    #              - 'down': a list of hidden state tensors from downsampling.
    #              - 'middle': the tensor of the output of the lowest-resolution
    #                          block in the model.
    #              - 'up': a list of hidden state tensors from upsampling.
    #     """
    #     hs = []
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)
    #     result = dict(down=[], up=[])
    #     h = x.type(self.inner_dtype)
    #     for module in self.input_blocks:
    #         h = module(h, emb)
    #         hs.append(h)
    #         result["down"].append(h.type(x.dtype))
    #     h = self.middle_block(h, emb)
    #     result["middle"] = h.type(x.dtype)
    #     for module in self.output_blocks:
    #         cat_in = th.cat([h, hs.pop()], dim=1)
    #         h = module(cat_in, emb)
    #         result["up"].append(h.type(x.dtype))
    #     return result

class UNetModel_cnn(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        #attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        #self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            #AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads_upsample,
                #         )
                #     )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result

class SuperResModel(UNetModel_cnn):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

if __name__ == "__main__":
    x=th.randn(1,3,32,32)
    aa=AttentionBlock(3)