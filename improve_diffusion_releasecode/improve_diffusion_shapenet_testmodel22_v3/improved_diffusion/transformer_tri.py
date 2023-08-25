import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import ipdb
# helpers
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
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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

class FeedForward_small(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
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

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
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

        dots = torch.matmul(query, key.transpose(-1, -2)) * self.scale  #1 16 65 65

        attn = self.attend(dots)
        attn = self.dropout(attn)
        #ipdb.set_trace()

        out = torch.matmul(attn, value)   #1 16 65 64
        out = rearrange(out, 'b h n d -> b n (h d)') #1  65 1024
        return self.to_out(out).view(b,3,-1,d)  #1  65 1024

class Attention_q(nn.Module):
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

    def forward(self, query,x):
        #ipdb.set_trace()
        b,_,n,d=x.shape
        query=self.query(query)
        key=self.key(x.view(b,-1,d))
        value=self.value(x.view(b,-1,d))
        
        key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [key,value])  #1 16 65 64

        ff=lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads)

        query=ff(query) 

        #ipdb.set_trace()

        dots = torch.matmul(query, key.transpose(-1, -2)) * self.scale  #1 16 65 65

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value)   #1 16 65 64
        out = rearrange(out, 'b h n d -> b n (h d)') #1  65 1024
        return self.to_out(out)  #1  65 1024

class Transformer_depend(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.encoder = nn.ModuleList([])
        for _ in range(depth):
            self.encoder.append(nn.ModuleList([
                PreNorm(dim, Attention3d(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        self.decoder = nn.ModuleList([])
        for _ in range(depth):

            self.decoder.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                Attention_q(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                #PreNorm(dim, nn.Identity()),
                PreNorm(dim, nn.Identity()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, query,x):
        #ipdb.set_trace()
        for attn, ff in self.encoder:
            x = attn(x) + x
            x = ff(x) + x
  
        for att_q,attn,norm2, ff in self.decoder:
            
            query=att_q(query)+query
            #ipdb.set_trace()
            query = norm2(attn(query,x) + query)
            #ipdb.set_trace()
            query = ff(query) + query
        return query

class Transtri(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 32, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        #ipdb.set_trace()
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding1 = nn.Conv2d(channels, dim, kernel_size=patch_height, stride=patch_height, bias=True)
        self.to_patch_embedding2 = nn.Conv2d(channels, dim, kernel_size=patch_height, stride=patch_height, bias=True)
        self.to_patch_embedding3 = nn.Conv2d(channels, dim, kernel_size=patch_height, stride=patch_height, bias=True)
        # nn.Sequential(
        #     Rearrange('b a c (h p1) (w p2) -> b a (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        #     SiLU(),
        #     nn.Linear(dim,dim),
        # )
        self.dropout = nn.Dropout(dropout)



        self.transformerx = Transformer_depend(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformery = Transformer_depend(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformerz = Transformer_depend(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pos_embedding=nn.Parameter(torch.zeros(1, 3,num_patches, dim))
        self.pool = pool
        self.weightpatchx=nn.Parameter(torch.ones(1))
        self.weightpatchy=nn.Parameter(torch.ones(1))
        self.weightpatchz=nn.Parameter(torch.ones(1))
        self.to_patch_embeddingx = nn.Sequential(
            nn.Linear(dim, patch_dim),
            SiLU(),
            nn.Linear(patch_dim,patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, w=(image_width // patch_width),h=(image_height // patch_height)),
        )
        self.to_patch_embeddingy = nn.Sequential(
            nn.Linear(dim, patch_dim),
            SiLU(),
            nn.Linear(patch_dim,patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, w=(image_width // patch_width),h=(image_height // patch_height)),
        )
        self.to_patch_embeddingz = nn.Sequential(
            nn.Linear(dim, patch_dim),
            SiLU(),
            nn.Linear(patch_dim,patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, w=(image_width // patch_width),h=(image_height // patch_height)),
        )


    def forward(self, img):
        #ipdb.set_trace()
        b, _, w,h = img.shape
        img=img.view(b,3,-1,w,h)
        x = torch.stack([self.to_patch_embedding1(img[:,0,:,:,:]).flatten(2).transpose(1, 2),\
                         self.to_patch_embedding2(img[:,1,:,:,:]).flatten(2).transpose(1, 2),\
                            self.to_patch_embedding3(img[:,2,:,:,:]).flatten(2).transpose(1, 2)],1)
        
        #ipdb.set_trace()
        #b, _, n,_ = x.shape

        #cls_tokens = repeat(self.cls_token, '1 1 1 d -> b 3 1 d', b = b)
        #ipdb.set_trace()
        #x = torch.cat((cls_tokens, x), dim=-2)
        #ipdb.set_trace()
        x += self.pos_embedding.repeat(b,1,1,1)
        x = self.dropout(x)
        #ipdb.set_trace()
        depx = self.weightpatchx*self.to_patch_embeddingx(self.transformerx(x[:,0,:,:],x))+img[:,0,:,:,:]
        depy =  self.weightpatchy*self.to_patch_embeddingy(self.transformery(x[:,1,:,:],x))+img[:,1,:,:,:]
        depz =  self.weightpatchz*self.to_patch_embeddingz(self.transformerz(x[:,2,:,:],x))+img[:,2,:,:,:]



        #ipdb.set_trace()

        return torch.cat([depx,depy,depz],1)


if __name__=='__main__':


    image_size = 256
    patch_size = 32
    dim = 32*32  #each patch feature vector
    transformer_channel=2048
    head=16
    transformer_numlayer=3



    model = ViTtri(
    image_size = image_size,
    patch_size = patch_size,
    dim = dim,
    depth = transformer_numlayer,
    heads = head,
    dim_head = transformer_channel//head,
    mlp_dim = dim*2,
    dropout = 0.1,
    emb_dropout = 0.1
    )
    

    img = torch.randn(1, 3, 32, 256, 256)
    #ipdb.set_trace()

    preds = model(img) # (1, 1000)