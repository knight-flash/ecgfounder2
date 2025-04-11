from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from vit1d import vit_base, vit_small, vit_tiny, vit_middle
from net1d import Net1D
from transformers import GPT2Model, GPT2Tokenizer

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(0, 2, 1) # convert X shape (B, C, L) to (B, L, C)

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1) 
        x = torch.cat((self.cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, :, :].to(x.dtype)  # (L+1)NC
        x, att_map = self.mhsa(x[:, :1, :], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        return x.squeeze(0), att_map[:, :, 1:]
    
class ECGCLIP(torch.nn.Module):
    def __init__(self, network_config):
        super(ECGCLIP, self).__init__()
        
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']
        
        # ECG信号编码器
        self.ecg_model = network_config['ecg_model']
        self.num_leads = network_config['num_leads']
        
        self.text_model = network_config['text_model']
    
        if 'resnet' in self.ecg_model:
            if self.ecg_model == 'resnet18':
                model = ResNet18()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                     embed_dim=self.proj_out, 
                                                     num_heads=4, 
                                                     output_dim=self.proj_out)
            elif self.ecg_model == 'resnet34':
                model = ResNet34()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                     embed_dim=self.proj_out, 
                                                     num_heads=4, 
                                                     output_dim=self.proj_out)
            elif self.ecg_model == 'resnet50':
                model = ResNet50()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                     embed_dim=self.proj_out, 
                                                     num_heads=4, 
                                                     output_dim=self.proj_out)
            elif self.ecg_model == 'resnet101':
                model = ResNet101()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                     embed_dim=self.proj_out, 
                                                     num_heads=4, 
                                                     output_dim=self.proj_out)
    
            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)
    
        if 'vit' in self.ecg_model:
            if self.ecg_model == 'vit_tiny':
                model = vit_tiny(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_small':
                model = vit_small(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_middle':
                model = vit_middle(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_base':
                model = vit_base(num_leads=self.num_leads)
            self.proj_e_input = model.width    
            self.proj_e = nn.Sequential(
                nn.Linear(self.proj_e_input, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )
            self.linear1 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)

        if 'ecgfounder' in self.ecg_model:
            model = Net1D(
                    in_channels=self.num_leads, 
                    base_filters=64, #32 64
                    ratio=1, 
                    filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
                    m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
                    kernel_size=16, 
                    stride=2, 
                    groups_width=16,
                    verbose=False, 
                    use_bn=False,
                    use_do=False,
                    return_features = True,
                    n_classes=self.proj_out)
            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)

    
        self.ecg_encoder = model
        self.avgpool = nn.AdaptiveAvgPool1d(1)
            
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        
        # freezing 12 layers
        freeze_layers = 6
        # 文本编码器（GPT2）
        if 'gpt' in self.text_model: 
            self.lm_model = GPT2Model.from_pretrained('/home/yanmingke/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
            self.tokenizer = GPT2Tokenizer.from_pretrained('/home/yanmingke/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置 pad_token 为 eos_token
            self.proj_t = nn.Sequential(
            nn.Linear(self.lm_model.config.hidden_size, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
            )
            for layer_idx in range(freeze_layers):
                for param in self.lm_model.h[layer_idx].parameters():
                    param.requires_grad = False
            
        # medcpt
        if 'medcpt' in self.text_model: 
            url = '/home/yanmingke/E-Zero/retrieval/medcpt'
            self.lm_model = AutoModel.from_pretrained(
                url)
            self.tokenizer = AutoTokenizer.from_pretrained(
                url)
            # 文本投影层
            self.proj_t = nn.Sequential(
                nn.Linear(self.lm_model.config.hidden_size, self.proj_hidden),
                nn.GELU(),
                nn.Linear(self.proj_hidden, self.proj_out),
            )
            # 针对 BERT
            for layer_idx in range(freeze_layers):
                for param in self.lm_model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False


    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        return tokenizer_output

    @torch.no_grad()
    def ext_ecg_emb(self, ecg):
        if 'resnet' in self.ecg_model:

            ecg_emb = self.ecg_encoder(ecg)
            
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb, att_map = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)
    
        if 'vit' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

        if 'ecgfounder' in self.ecg_model:
            # attention pooling (only for resnet models)
            ecg_emb,proj_ecg_emb = self.ecg_encoder(ecg)

        return proj_ecg_emb
    
    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        outputs = self.lm_model(input_ids=input_ids,
                                attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 使用 attention_mask 进行平均池化
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        text_emb = sum_embeddings / sum_mask
        return text_emb
        
    def forward(self, ecg, input_ids, attention_mask):
    
        if 'resnet' in self.ecg_model:
            # attention pooling (only for resnet models)
            ecg_emb = self.ecg_encoder(ecg)
            #print('ecg_emb shape:', ecg_emb.shape)
            ecg_emb = self.downconv(ecg_emb)
            proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)
    
            ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))
        
        if 'vit' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))

        if 'ecgfounder' in self.ecg_model:
            # attention pooling (only for resnet models)
            ecg_emb,proj_ecg_emb = self.ecg_encoder(ecg)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))
    
        proj_ecg_emb = nn.functional.normalize(proj_ecg_emb, dim=-1)
    
        # 获取文本特征
        with torch.no_grad():
            text_emb = self.get_text_emb(input_ids, attention_mask)
        # text_emb = self.get_text_emb(input_ids, attention_mask)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = nn.functional.normalize(proj_text_emb, dim=-1)
    
        if self.training:
            return {'ecg_emb': [ecg_emb1, ecg_emb2],
                    'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb]}
        else:
            return {'ecg_emb': [ecg_emb1, ecg_emb2],
                    'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb]}