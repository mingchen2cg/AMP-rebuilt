import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoModel

class ProstT5WithGatedFusion(nn.Module):
    def __init__(self, t2struc_text_encoder, prostt5_model, hidden_dim=1024, text_dim=768):
        super().__init__()
        
        # 1. 冻结的 Text Encoder (来自 T2Struc)
        self.text_encoder = t2struc_text_encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # 2. 主模型 ProstT5 (Encoder + Decoder)
        # 注意：这里我们传入整个 model，但在 forward 里会拆开用
        self.prostt5 = prostt5_model
        self.hidden_dim = hidden_dim # ProstT5 的 hidden_size (通常 1024)
        
        # 3. 适配器 (Projector)
        # 把文本特征维度 (如 768) 映射到 ProstT5 维度 (1024)
        self.text_projector = nn.Linear(text_dim, hidden_dim)
        
        # 4. 关键组件：Cross-Attention (用于对齐)
        # Query: 结构特征, Key/Value: 文本特征
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim) # 稳定性归一化
        
        # 5. 关键组件：自适应门控 (Adaptive Gate)
        # 输入是 [原始结构; 对齐后的文本]，输出是一个 0-1 的标量权重
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 降维融合
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),              # 输出每个 Token 的门控值
            nn.Sigmoid()                           # 归一化到 (0, 1)
        )

    def forward(self, 
                structure_input_ids, 
                structure_attention_mask,  # <--- 确保这里是 attention_mask，不是 mask
                text_input_ids, 
                text_attention_mask,       # <--- 确保这里是 attention_mask，不是 mask
                labels=None,
                **kwargs):                 # <--- 建议加上 **kwargs 以吸收可能的额外参数（如 return_dict 等）
        
        # --- A. 获取文本特征 (Frozen) ---
        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=text_input_ids, 
                attention_mask=text_attention_mask # <--- 这里也要对应修改
            )
            text_hidden = text_outputs.last_hidden_state 

        # --- B. 获取结构特征 (ProstT5 Encoder) ---
        struc_outputs = self.prostt5.encoder(
            input_ids=structure_input_ids, 
            attention_mask=structure_attention_mask # <--- 这里也要对应修改
        )
        struc_hidden = struc_outputs.last_hidden_state

        # --- C. 投影与对齐 (Alignment) ---
        text_proj = self.text_projector(text_hidden) 
        
        # mask 是 1 表示有效，0 表示 padding 
        # key_padding_mask 需要 True 表示 padding (无效)
        key_padding_mask = (text_attention_mask == 0) # <--- 这里也要对应修改

        context, _ = self.cross_attn(
            query=struc_hidden,
            key=text_proj,
            value=text_proj,
            key_padding_mask=key_padding_mask
        )
        context = self.attn_norm(context)

        # --- D. 门控融合 (Gated Fusion) ---
        concat_features = torch.cat([struc_hidden, context], dim=-1)
        gate = self.gate_net(concat_features)
        fused_hidden = (1 - gate) * struc_hidden + gate * context

        # --- E. 解码 (ProstT5 Decoder) ---
        outputs = self.prostt5(
            encoder_outputs=(fused_hidden,), 
            attention_mask=structure_attention_mask, # <--- 这里也要对应修改
            labels=labels,
            **kwargs # 传递剩余参数
        )

        return outputs