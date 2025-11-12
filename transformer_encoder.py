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
    def __init__(self, 
                 input_dim=13,  # Updated default: 12 motion features + 1 confidence
                 d_model=256,   # Increased from 128 for more capacity
                 nhead=8,
                 num_layers=6,  # Increased from 3 for complex motion patterns
                 dim_ff=1024,   # Increased from 512
                 dropout=0.1,
                 use_residual_prediction=True,  # Predict residuals from last known position
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_residual_prediction = use_residual_prediction
        
        # Deeper input embedding network
        self.in_fc = nn.Sequential(
            nn.Linear(input_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model)
        )
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=dim_ff,
                batch_first=True,
                dropout=dropout,
                activation='gelu',
            ),
            mask_check=False,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        # Deeper output network
        self.out_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, input_dim)
        )

    @staticmethod
    def _mask(src_size, trg_size, device):
        mask = torch.triu(torch.ones(src_size + trg_size, src_size + trg_size, device=device), diagonal=1)
        mask[:, :src_size] = 0
        return mask.bool()

    def forward(self, src, trg):
        input_tensor = torch.cat([src, trg], dim=1)
        enc_emb = self.pos_enc(self.in_fc(input_tensor) * math.sqrt(self.d_model))
        mask = self._mask(src.size(1), trg.size(1), input_tensor.device)
        out = self.transformer.forward(enc_emb, mask=mask)
        pred_offset = self.out_fc(out[:, -trg.size(1):, :])
        
        if self.use_residual_prediction:
            # Predict offset from last known position for better generalization
            # Use only the bbox coordinates (first 4 dims) from the last source frame
            last_src_bbox = src[:, -1:, :4]  
            pred_bbox = pred_offset[:, :, :4] + last_src_bbox
            # Combine predicted bbox with predicted motion features and confidence
            pred_full = torch.cat([pred_bbox, pred_offset[:, :, 4:]], dim=2)
            return pred_full
        else:
            return pred_offset + trg

    @torch.no_grad()
    def inference(self, src, trg, num_steps=1):
        preds = []
        for t in range(num_steps):
            trg_ = torch.cat([trg[:, 0:1, :]] + preds, dim=1)
            out = self.forward(src, trg_)
            preds.append(out[:, -1:, :])
        return torch.cat(preds, dim=1)
    
    def train_one_epoch(self, dataloader, optimizer, criterion, device='cuda'):
        self.train()
        total_loss = 0

        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            output = self.forward(src, trg[:, :-1])

            loss = criterion(output, trg[:, 1:])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
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