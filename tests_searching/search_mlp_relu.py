import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config_py import *
    


class PatentDenoiserMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 activation: str = 'gelu', dropout: float = 0.1):
        super().__init__()
        
        self.activation = self._get_activation(activation)
        
        # Базовая сеть: обучает нелинейную "поправку" Δ
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Инициализация: последний слой ≈ 0, чтобы на старте model(x) ≈ x
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)
        
    def _get_activation(self, name: str) -> nn.Module:
        acts = {
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh()
        }
        if name not in acts:
            raise ValueError(f"Unknown activation: {name}. Use {list(acts.keys())}")
        return acts[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim], уже L2-нормализован
        Возвращает: [batch_size, input_dim], L2-нормализованный
        """
        delta = self.delta_net(x)          # учим шум/коррекцию
        out = x + delta                    # остаточное соединение
        return F.normalize(out, p=2, dim=-1) # возврат на единичную сферу


def transform_query_vector(vector): 

    model = PatentDenoiserMLP(
        input_dim=len(vector), 
        hidden_dim=len(vector)//2, 
        activation='relu', 
        dropout=0.1
    )

    checkpoint = torch.load(CHECKPOINT_RELU, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    vector = torch.from_numpy(vector).float()

    if vector.dim() == 1:
        vector = vector.unsqueeze(0)

    with torch.no_grad():
        cleaned_vector = model(vector)

    query_vec_np = cleaned_vector.cpu().numpy().flatten() 

    return query_vec_np