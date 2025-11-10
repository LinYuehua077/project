import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """时间序列的Patch Embedding层"""
    
    def __init__(self, seq_len, patch_size, d_model, in_channels=1):
        """
        输入参数:
            seq_len: 输入序列长度
            patch_size: 每个patch包含的时间步数
            d_model: 嵌入维度
            in_channels: 输入特征通道数
        """
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_channels = in_channels
        
        # 计算patch数量
        self.num_patches = seq_len // patch_size
        if self.num_patches * patch_size != seq_len:
            raise ValueError(f"序列长度{seq_len}必须能被patch大小{patch_size}整除")
        
        # Patch embedding层
        self.patch_embed = nn.Linear(patch_size * in_channels, d_model)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, in_channels]
        Returns:
            embedded_patches: 嵌入后的patch序列 [batch_size, num_patches, d_model]
        """
        batch_size = x.shape[0]
        
        # 重塑为patches [batch_size, num_patches, patch_size * in_channels]
        x = x.reshape(batch_size, self.num_patches, self.patch_size * self.in_channels)
        
        # Patch embedding
        embedded_patches = self.patch_embed(x)  # [batch_size, num_patches, d_model]
        
        # 添加位置编码
        embedded_patches = embedded_patches + self.pos_embedding
        
        return embedded_patches