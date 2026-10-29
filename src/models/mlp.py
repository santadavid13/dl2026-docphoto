"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3, prior: float = 0.5):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        
        # Rejtett rétegek (Linear + ReLU + Dropout) felépítése a `hidden` lista alapján
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
            
        self.features = nn.Sequential(*layers)
        
        # Kimeneti réteg (1 db kimenet)
        self.head = nn.Linear(prev_dim, 1)
        
        # Trükk a 'prior' használatára: 
        # Mivel a végén Sigmoidot használunk, a kezdeti bias-t úgy állítjuk be,
        # hogy a hálózat alapértelmezetten a betanító halmaz átlagát (prior) jósolja.
        # Ennek az értéke az inverz sigmoid, azaz: log(prior / (1 - prior))
        bias_val = math.log(prior / (1.0 - prior))
        nn.init.constant_(self.head.bias, bias_val)
        nn.init.zeros_(self.head.weight) # Súlyok nullázása a tiszta prior induláshoz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        # Sigmoid szorítja be az eredményt a [0, 1] intervallumba
        x = torch.sigmoid(x)
        # Squeeze eltávolítja az utolsó dimenziót, így (batch, 1) helyett (batch,) lesz a forma
        return x.squeeze(-1)