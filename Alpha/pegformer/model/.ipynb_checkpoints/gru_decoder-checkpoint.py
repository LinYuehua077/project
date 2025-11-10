import torch
import torch.nn as nn
from typing import Optional


class GRUDecoder(nn.Module):
    """GRU解码器模块"""
    
    def __init__(self, d_model, hidden_size, output_size, num_layers=2, dropout=0.1):
        """
        输入参数:
            d_model: 输入维度(此处应等于Transformer编码器的输出维度)
            hidden_size: GRU隐藏层大小
            output_size: 输出维度
            num_layers: GRU层数
            dropout: dropout率
        """
        super().__init__()
        
        # 参数初始化
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
        
    def forward(self, x, target_len):
        """
        输入参数:
            x: 编码器输出 [batch_size, num_patches, d_model]
            target_len: 目标序列长度，如果为None则与输入序列长度相同
        输出值:
            output: 解码输出 [batch_size, target_len, output_size]
        """
        batch_size, seq_len, _ = x.shape
        target_len = target_len or seq_len
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # GRU前向传播
        gru_out, _ = self.gru(x, h0)  # [batch_size, seq_len, hidden_size]
        
        # 如果目标长度与输入不同，我们需要进行序列生成
        if target_len != seq_len:
            # 使用最后一个时间步的隐藏状态来生成更长的序列
            last_hidden = gru_out[:, -1:, :]  # [batch_size, 1, hidden_size]
            
            # 递归生成序列
            outputs = []
            current_input = last_hidden
            hidden_state = h0
            
            for _ in range(target_len):
                gru_out_step, hidden_state = self.gru(current_input, hidden_state)
                output_step = self.output_layer(gru_out_step)
                outputs.append(output_step)
                current_input = gru_out_step
            
            output = torch.cat(outputs, dim=1)  # [batch_size, target_len, output_size]
        else:
            # 直接输出
            output = self.output_layer(gru_out)  # [batch_size, seq_len, output_size]
        
        return output