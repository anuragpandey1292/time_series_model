"""
Industry-grade LSTM model for time series forecasting
Model definition ONLY
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    horizon: int = 1
    bidirectional: bool = False


class LSTMModel(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()

        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        multiplier = 2 if config.bidirectional else 1

        self.fc = nn.Linear(
            config.hidden_size * multiplier,
            config.horizon
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:

        out, _ = self.lstm(x, hidden)
        last_out = out[:, -1, :]
        return self.fc(last_out)
