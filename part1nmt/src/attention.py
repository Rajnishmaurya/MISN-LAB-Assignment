import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def project_encoder(self, encoder_outputs):
        """Precompute W2(encoder_outputs) once per batch to avoid repeating it at every decoder step."""
        return self.W2(encoder_outputs)

    def forward(self, hidden, encoder_outputs, encoder_projection=None):
        hidden = hidden.unsqueeze(1)

        if encoder_projection is None:
            encoder_projection = self.W2(encoder_outputs)

        score = self.V(
            torch.tanh(
                self.W1(hidden) + encoder_projection
            )
        )

        attention_weights = torch.softmax(score, dim=1)

        context = attention_weights * encoder_outputs
        context = context.sum(dim=1)

        return context, attention_weights
