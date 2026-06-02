import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
    
from config_py import *

class PatentAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 896, bottleneck_dim: int = 256, 
                 hidden_dim: int = 512, activation: str = 'gelu'):
        super().__init__()
        
        # Выбор функции активации
        act_func = nn.ReLU() if activation == 'gelu' else nn.SiLU()
        
        # --- ENCODER (Сжатие) ---
        # d=896 -> h=512 -> p=256
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_func,
            nn.Dropout(0.1),  # Регуляризация
            nn.Linear(hidden_dim, bottleneck_dim),
            act_func          # Активация перед узким местом
        )
        
        # --- DECODER (Восстановление) ---
        # p=256 -> h=512 -> d=896
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            act_func,
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
            # Выходной слой БЕЗ активации, нормализуем позже
        )
        
        # Инициализация последнего слоя в ноль для стабильного старта
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 896]
        return: [batch, 896] — восстановленный, очищенный вектор
        """
        # 1. Сжатие в латентное пространство (z)
        z = self.encoder(x)
        
        # 2. Восстановление в исходное пространство (x_hat)
        x_hat = self.decoder(z)
        
        # 3. L2-нормализация (КРИТИЧНО для косинусного поиска)
        return F.normalize(x_hat, p=2, dim=-1)


def transform_query_vector(vector): 

    model = PatentAutoencoder(
        input_dim=len(vector), 
        hidden_dim=512, 
        activation='gelu', 
    )

    checkpoint = torch.load(CHECKPOINT_AE_RELU, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    vector = torch.from_numpy(vector).float()

    if vector.dim() == 1:
        vector = vector.unsqueeze(0)

    with torch.no_grad():
        cleaned_vector = model(vector)

    query_vec_np = cleaned_vector.cpu().numpy().flatten() 

    return query_vec_np