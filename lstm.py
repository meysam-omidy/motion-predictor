import torch
from torch import nn
import random
import os

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=4, middle_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.fc_in = nn.Linear(input_dim, middle_dim, dtype=torch.float32)
        self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float32)
        self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float32)
        self.fc_out = nn.Linear(hidden_dim, input_dim, dtype=torch.float32)  # predict offset (dx, dy, dw, dh)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        # src: (batch, seq_len, 4)
        # trg: (batch, seq_len, 4) - ground truth future sequence
        outputs = []

        batch_size, trg_size, _ = trg.size()
        # Encode
        _, (hidden, cell) = self.encoder(self.fc_in(src))

        # First input to decoder is the last frame of src
        decoder_input = trg[:, 0:1, :]  # shape (batch, 1, 4)


        for t in range(1, trg_size + 1):
            out, (hidden, cell) = self.decoder(self.fc_in(decoder_input), (hidden, cell))
            pred = self.fc_out(out)  # (batch, 1, 4)
            outputs.append(pred)

            # Decide if we use teacher forcing
            if t != trg_size:
                use_teacher = trg is not None and random.random() < teacher_forcing_ratio
                decoder_input = trg[:, t:t+1, :] if use_teacher else (pred + trg[:, t-1:t, :])

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, 4)
        return outputs + trg
    
    def inference(self, src, trg, num_steps=1):
        outputs = []
        # Encode
        _, (hidden, cell) = self.encoder(self.fc_in(src))

        # First input to decoder is the last frame of src
        decoder_input = trg[:, 0:1, :]  # shape (batch, 1, 4)


        for t in range(num_steps):
            out, (hidden, cell) = self.decoder(self.fc_in(decoder_input), (hidden, cell))
            pred = self.fc_out(out)  # (batch, 1, 4)
            decoder_input = pred + decoder_input
            outputs.append(decoder_input)

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, 4)
        return outputs


    def train_one_epoch(self, dataloader, optimizer, criterion, teacher_forcing_ratio=0.5, device='cuda'):
        self.train()
        total_loss = 0

        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            output = self.forward(src, trg[:, :-1], teacher_forcing_ratio)

            loss = criterion(output, trg[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def evaluate(self, dataloader, criterion, device='cuda'):
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for src, trg in dataloader:
                src = src.to(device)
                trg = trg.to(device)
                output = self.inference(src, trg, num_steps=trg.size(1) - 1)
                loss = criterion(output, trg[:, 1:])
                total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def save_weight(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path):
        self.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))