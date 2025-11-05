import math, torch, torch.nn as nn
import torch.nn.functional as F
import random
import os

# ---------- positional encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                       # (1,L,D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---------- encoder–decoder transformer ----------
class MotionTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=128, nhead=8,
                 num_enc_layers=3, num_dec_layers=3,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.in_fc = nn.Sequential(
            nn.Linear(input_dim, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        # self.in_fc = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_ff,
                                          dropout=dropout, batch_first=True)
        self.out_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, input_dim)
        )
        # self.out_fc = nn.Linear(d_model, input_dim)  # predicts offset Δ

    def encode(self, src):
        src_emb = self.pos_enc(self.in_fc(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb)
        return memory

    def decode_step(self, memory, dec_in):
        # dec_in: (B,t,4)
        dec_emb = self.pos_enc(self.in_fc(dec_in) * math.sqrt(self.d_model))
        tgt_mask = self._causal_mask(dec_emb.size(1), device=dec_emb.device)
        out = self.transformer.decoder(dec_emb, memory, tgt_mask=tgt_mask)
        pred_offset = self.out_fc(out[:, -1:, :])     # predict Δ for last step
        return pred_offset

    @staticmethod
    def _causal_mask(size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    # -------- training forward pass with teacher forcing --------
    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        """
        src: (B,S,4) observed boxes
        trg: (B,T,4) future *absolute* boxes
        returns predicted *absolute* boxes for the whole target horizon
        """
        B, T, _ = trg.shape
        memory = self.encode(src)

        # start with the last observed box as initial decoder input
        prev_box = trg[:, 0:1, :]
        # prev_box = src[:, -1:, :]
        preds = []

        for t in range(T):
            offset = self.decode_step(memory, prev_box)

            # compute next box as prev_box + predicted offset
            next_box = trg[:, t:t+1, :] + offset
            # if t == 0:
            #     next_box = src[:, -1:, :] + offset
            # else:
            #     next_box = trg[:, t-1:t, :] + offset
            # next_box = prev_box[:, -1:, :] + offset
            preds.append(next_box)

            if t < T-1:
                # choose next decoder input: ground truth or model prediction
                use_teacher = (random.random() < teacher_forcing_ratio)
                next_in = trg[:, t+1:t+2, :] if use_teacher else next_box
                # next_in = trg[:, t:t+1, :] if use_teacher else next_box
                prev_box = torch.cat([prev_box, next_in], dim=1)

        return torch.cat(preds, dim=1)   # (B,T,4)

    # -------- inference rollout (no teacher forcing) --------
    @torch.no_grad()
    def inference(self, src, trg, num_steps=1):
        memory = self.encode(src)
        prev_box = trg[:, 0:1, :]
        preds = []
        for _ in range(num_steps):
            offset = self.decode_step(memory, prev_box)
            next_box = prev_box[:, -1:, :] + offset
            preds.append(next_box)
            prev_box = torch.cat([prev_box, next_box], dim=1)
        return torch.cat(preds, dim=1)
    
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