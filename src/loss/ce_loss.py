import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    Wrapper over PyTorch CrossEntropyLoss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            Tensor: (batch, seq_len, hidden_dim)
            Tensor: (batch, seq_len)

        Returns:
            loss (dict): dict containing calculated loss functions.
        """

        return {
            "loss": self.loss(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        }
