import torch
import torch.nn as nn
from typing import Optional

from .patch_embedding import PatchEmbedding
from .transformer_encoder import TransformerEncoder
from .gru_decoder import GRUDecoder


class PegFormer(nn.Module):
    """PegFormer模型 - 结合Transformer编码器和GRU解码器的时间序列预测模型"""
    
    def __init__(self, seq_len, patch_size, d_model, nhead, 
                 num_encoder_layers, hidden_size, num_decoder_layers,
                 output_size, in_channels=1, dim_feedforward=2048,
                 dropout=0.1):
        """
        输入参数:
            seq_len: 输入序列长度
            patch_size: patch大小
            d_model: 模型维度
            nhead: Transformer头数
            num_encoder_layers: Transformer编码器层数
            hidden_size: GRU隐藏层大小
            num_decoder_layers: GRU层数
            output_size: 输出特征维度
            in_channels: 输入特征维度
            dim_feedforward: Transformer前馈网络维度
            dropout: dropout率
        """
        super().__init__()
        
        # 初始化模型的参数
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.d_model = d_model
        self.output_size = output_size
        
        # 初始化模型的模块
        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(
            seq_len=seq_len,
            patch_size=patch_size,
            d_model=d_model,
            in_channels=in_channels
        )
        
        # 2. Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 3. GRU Decoder
        self.gru_decoder = GRUDecoder(
            d_model=d_model,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, target_len):
        """
        输入参数:
            x: 输入序列 [batch_size, seq_len, in_channels]
            target_len: 目标预测长度
        返回值:
            output: 预测序列 [batch_size, target_len, output_size]
        """
        # 1. Patch Embedding
        patches = self.patch_embedding(x)  # [batch_size, num_patches, d_model]
        patches = self.layer_norm1(patches)
        
        # 2. Transformer Encoder
        encoded = self.transformer_encoder(patches)  # [batch_size, num_patches, d_model]
        encoded = self.layer_norm2(encoded)
        
        # 3. GRU Decoder
        # 计算目标patch数量
        num_patches = self.patch_embedding.num_patches
        if target_len is not None:
            target_patches = (target_len + self.patch_size - 1) // self.patch_size
        else:
            target_patches = num_patches
        
        output = self.gru_decoder(encoded, target_patches)  # [batch_size, target_patches, output_size]
        
        # 如果需要，将patch序列转换回时间步序列
        if target_len is not None:
            # 重塑为时间步序列
            output = output.reshape(output.shape[0], -1, self.output_size)
            # 截断或填充到目标长度
            if output.shape[1] > target_len:
                output = output[:, :target_len, :]
            elif output.shape[1] < target_len:
                padding = torch.zeros(output.shape[0], target_len - output.shape[1], 
                                    self.output_size, device=output.device)
                output = torch.cat([output, padding], dim=1)
        
        return output


class MultiHeadPegFormer(nn.Module):
    """多头PegFormer，用于多变量时间序列预测"""
    
    def __init__(self, seq_len: int, patch_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, hidden_size: int, num_decoder_layers: int,
                 output_size: int, in_channels: int = 1, **kwargs):
        """
        Args:
            output_size: 输出变量数量
            其他参数同PegFormer
        """
        super().__init__()
        
        self.output_size = output_size
        
        # 为每个输出头创建单独的PegFormer
        self.heads = nn.ModuleList([
            PegFormer(
                seq_len=seq_len,
                patch_size=patch_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                hidden_size=hidden_size,
                num_decoder_layers=num_decoder_layers,
                output_size=1,  # 每个头输出1维
                in_channels=in_channels,
                **kwargs
            ) for _ in range(output_size)
        ])
        
    def forward(self, x: torch.Tensor, target_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [batch_size, seq_len, in_channels]
            target_len: 目标预测长度
        Returns:
            output: 预测序列 [batch_size, target_len, output_size]
        """
        outputs = []
        for head in self.heads:
            head_output = head(x, target_len)  # [batch_size, target_len, 1]
            outputs.append(head_output)
        
        # 拼接所有头的输出
        output = torch.cat(outputs, dim=-1)  # [batch_size, target_len, output_size]
        return output