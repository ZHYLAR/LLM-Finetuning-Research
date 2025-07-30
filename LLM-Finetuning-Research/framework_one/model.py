# model.py

import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA层，用于包装一个线性层并对其进行低秩适应。
    """
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.original_layer = original_layer
        d_in, d_out = original_layer.in_features, original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(d_in, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = alpha / rank
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块。
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.d_k = config.d_model // config.n_head
        self.n_head = config.n_head
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        y = (self.dropout(attn_weights)) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(y)

class FeedForward(nn.Module):
    """
    前馈神经网络模块。
    """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff), 
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model), 
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    单个Transformer块，包含多头注意力和前馈网络。
    """
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class ToyLLM(nn.Module):
    """
    完整的玩具大语言模型。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss