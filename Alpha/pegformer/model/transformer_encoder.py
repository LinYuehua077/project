import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Transformer编码器模块"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: dropout率
        """
        super().__init__()
        self.d_model = d_model
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 输入格式为 [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [batch_size, num_patches, d_model]
        Returns:
            encoded_seq: 编码后的序列 [batch_size, num_patches, d_model]
        """
        # 对于编码器，我们不需要causal mask，因为要看到整个序列
        # 如果需要处理可变长度序列，可以在这里添加padding mask
        encoded_seq = self.transformer_encoder(x)
        return encoded_seq