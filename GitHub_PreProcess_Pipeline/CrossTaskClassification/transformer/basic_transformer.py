import torch
import torch.nn as nn

# This baseline assumes:
# - each sample is one MU-band trial
# - shape = (batch_size, n_channels, input_dim)
# - output = state logits

class TimeSeriesTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_channels=256,
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=3,
        dropout=0.1,
    ):
        super().__init__()

        self.n_channels = n_channels

        # Project per-channel features to the transformer dimension.
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable channel embeddings match the MU feature layout used by the dataset.
        self.channel_embedding = nn.Parameter(torch.zeros(1, n_channels, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x: (batch_size, n_channels, input_dim)
        """
        _, seq_len, _ = x.shape

        if seq_len != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {seq_len}."
            )

        x = self.input_proj(x)
        x = x + self.channel_embedding

        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
