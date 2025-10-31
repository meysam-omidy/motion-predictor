import torch
from torch import nn
import random
import os

class ImprovedLSTMPredictor(nn.Module):
    def __init__(self, input_dim=4, middle_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        # self.fc_in = nn.Linear(input_dim, middle_dim)
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, middle_dim // 4),
            nn.GELU(),
            nn.Linear(middle_dim // 4, middle_dim // 2),
            nn.GELU(),
            nn.Linear(middle_dim // 2, middle_dim)
        )
        if num_layers > 1:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_dim*2, middle_dim)
        self.attn = nn.MultiheadAttention(middle_dim, num_heads=4, batch_first=True)
        if num_layers > 1:
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True)
        # self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, middle_dim // 2),
            nn.GELU(),
            nn.Linear(middle_dim // 2, middle_dim // 4),
            nn.GELU(),
            nn.Linear(middle_dim // 4, input_dim)
        )

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size, _, _ = src.size()
        _, trg_size, _ = trg.size()
        enc_embed = self.fc_in(src)
        enc_out, (h, c) = self.encoder(enc_embed)
        # Combine bidirectional states
        h = h.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        c = c.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        # Project encoder outputs for attention
        proj_enc = torch.tanh(self.attn_fc(enc_out))  # (batch, seq, middle_dim)

        prev_box = trg[:, 0:1, :]
        outputs = []
        for t in range(1, trg_size + 1):
            inp = self.fc_in(prev_box)
            # built-in multi-head attent ion
            attn_out, _ = self.attn(inp, proj_enc, proj_enc)
            dec_input = inp + attn_out  # residual
            out, (h, c) = self.decoder(dec_input, (h, c))
            delta = self.fc_out(out)
            # pred = prev_box + delta  # residual update
            pred = trg[:, t-1:t, :] + delta
            outputs.append(pred)
            if t < trg_size:
                use_teacher = trg is not None and random.random() < teacher_forcing_ratio
                prev_box = trg[:, t:t+1, :] if use_teacher else pred
        return torch.cat(outputs, dim=1)

    @torch.no_grad
    def inference(self, src, trg, num_steps=1):
        batch_size, _, _ = src.size()
        enc_embed = self.fc_in(src)
        enc_out, (h, c) = self.encoder(enc_embed)
        # Combine bidirectional states
        h = h.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        c = c.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        proj_enc = torch.tanh(self.attn_fc(enc_out))

        prev_box = trg[:, 0:1, :]
        outputs = []
        for _ in range(num_steps):
            inp = self.fc_in(prev_box)
            attn_out, _ = self.attn(inp, proj_enc, proj_enc)
            dec_input = inp + attn_out
            # out, _ = self.decoder(dec_input, (h, c))
            out, (h, c) = self.decoder(dec_input, (h, c))
            delta = self.fc_out(out)
            pred = prev_box + delta
            outputs.append(pred)
            prev_box = pred
        return torch.cat(outputs, dim=1)


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