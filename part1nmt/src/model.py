import torch
import torch.nn as nn
from attention import BahdanauAttention

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(emb)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.attention = BahdanauAttention(hidden_dim)

        self.lstm = nn.LSTM(
            emb_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs, encoder_projection=None):

        x = x.unsqueeze(1)

        emb = self.embedding(x)

        context, attention = self.attention(hidden[-1], encoder_outputs, encoder_projection)

        context = context.unsqueeze(1)

        lstm_input = torch.cat([emb, context], dim=-1)

        outputs, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        logits = self.fc(outputs.squeeze(1))

        return logits, hidden, cell, attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        encoder_projection = self.decoder.attention.project_encoder(encoder_outputs)

        x = tgt[:, 0]

        for t in range(1, tgt_len):

            output, hidden, cell, _ = self.decoder(
                x,
                hidden,
                cell,
                encoder_outputs,
                encoder_projection
            )

            outputs[:, t] = output

            best_guess = output.argmax(1)

            x = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else best_guess

        return outputs
