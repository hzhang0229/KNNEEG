from transformers import ViTModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTAttention
import transformers
import torch
import torch.nn as nn

"""
This is the initial KNNViT model, which proves its capability to be applied on ET problems. 
Note that this model will be changed later for better performance.
"""

class kNNAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,topk=100):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.topk = topk
    
    def forward(self, x, output_attentions=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # the core code block
        mask=torch.zeros(B,self.num_heads,N,N,device=x.device,requires_grad=False)
        index=torch.topk(attn,k=self.topk,dim=-1,largest=True)[1]
        mask.scatter_(-1,index,1.)
        attn=torch.where(mask>0,attn,torch.full_like(attn,float('-inf')))
        # end of the core code block

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
        


class KNNEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        self.config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=256,
            image_size=(129,14),
            patch_size=(8,1)
        )
        self.model = ViTModel(self.config)

        #####################################################
        # Replace the attention layers in the encoder
        for block in self.model.encoder.layer:
            # Assuming each block has an 'attention' attribute
            knn_attention = kNNAttention(
                dim=768,
                num_heads=12,
                qkv_bias=False,  # Set this according to your needs
                qk_scale=None,  # This could be set as config.qk_scale if such an attribute exists
                attn_drop=self.config.attention_probs_dropout_prob,
                proj_drop=self.config.hidden_dropout_prob,
                topk= 100
            )
            block.attention.self = knn_attention
        #####################################################


        model = ViTModel(self.config)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                                       torch.nn.Linear(768,2,bias=True))
        self.ViT = model
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT(x).pooler_output
        
        return x
    



    