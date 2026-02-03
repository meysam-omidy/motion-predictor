from torch import nn
import torch
import torch.nn.functional as F
import random
import os


class ImprovedLSTMPredictor(nn.Module):
    """
    LSTM-based motion predictor with the same API as MotionTransformer:
    - input_dim=13 (12 motion features + 1 confidence), output_dim=5 (4 bbox + 1 confidence)
    - forward(src, trg): trg is target context [B, T, 13], returns predictions [B, T, 5] for next steps
    - Uses residual bbox updates and sigmoid confidence like the transformer.
    """

    def __init__(
        self,
        input_dim=13,
        output_dim=5,
        d_model=256,
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Input embedding (same style as transformer)
        self.in_fc = nn.Sequential(
            nn.Linear(input_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
        )

        # Encoder: bidirectional LSTM over source sequence
        self.encoder = nn.LSTM(
            d_model,
            hidden_dim,
            num_encoder_layers,
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0,
            bidirectional=True,
        )
        self.enc_hidden_dim = hidden_dim
        self.attn_fc = nn.Linear(hidden_dim * 2, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(
            d_model,
            hidden_dim,
            num_decoder_layers,
            batch_first=True,
            dropout=dropout if num_decoder_layers > 1 else 0,
        )

        # Output head (same style as transformer: 5 dims -> bbox residual + confidence logit)
        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, output_dim),
        )

    def _merge_bidirectional(self, h, num_layers):
        # h: (num_layers * 2, B, hidden) -> (num_layers, B, hidden*2) then sum -> (num_layers, B, hidden)
        batch_size = h.size(1)
        hidden_size = h.size(2)
        h = h.view(num_layers, 2, batch_size, hidden_size).sum(dim=1)
        return h

    def forward(self, src, trg, teacher_forcing_ratio=None):
        """
        src: (B, S, input_dim), trg: (B, T, input_dim) target context (e.g. trg[:, :-1] in training).
        Returns: (B, T, output_dim) with bbox as residual (prev + delta) and confidence in [0,1].
        """
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        batch_size, src_len, _ = src.size()
        _, trg_len, _ = trg.size()

        # Encode source
        enc_embed = self.in_fc(src)
        enc_out, (h_enc, c_enc) = self.encoder(enc_embed)
        h_enc = self._merge_bidirectional(h_enc, self.encoder.num_layers)
        c_enc = self._merge_bidirectional(c_enc, self.encoder.num_layers)
        proj_enc = torch.tanh(self.attn_fc(enc_out))

        # Decode step-by-step
        prev = trg[:, 0:1, :]  # (B, 1, 13) — use first 5 for "previous bbox+conf" when applying residual
        outputs = []
        for t in range(trg_len):
            inp_embed = self.in_fc(prev)
            attn_out, _ = self.attn(inp_embed, proj_enc, proj_enc)
            dec_input = inp_embed + attn_out
            out, (h_enc, c_enc) = self.decoder(dec_input, (h_enc, c_enc))
            raw = self.out_fc(out)  # (B, 1, 5)
            # Residual bbox + sigmoid confidence (same as transformer)
            pred_bbox = prev[:, :, :4] + raw[:, :, :4]
            pred_conf = torch.sigmoid(raw[:, :, 4:5])
            pred = torch.cat([pred_bbox, pred_conf], dim=-1)
            outputs.append(pred)
            if t + 1 < trg_len:
                use_teacher = random.random() < teacher_forcing_ratio
                next_gt = trg[:, t + 1 : t + 2, :]
                # For teacher forcing we need to feed next ground-truth *input* (13-dim);
                # we only have predicted 5-dim, so we copy predicted bbox+conf into first 5 of next input
                if use_teacher:
                    prev = next_gt
                else:
                    # Build 13-dim input from 5-dim pred: bbox (0:4), conf (12); zero motion (4:12)
                    next_in = prev.clone()
                    next_in[:, :, :4] = pred[:, :, :4]
                    next_in[:, :, 12:13] = pred[:, :, 4:5]
                    next_in[:, :, 4:12] = 0
                    prev = next_in
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def inference(self, src, trg, num_steps=1):
        """
        Autoregressive inference. trg should contain at least the first step (B, 1, 13).
        Returns (B, num_steps, 5).
        """
        batch_size = src.size(0)
        enc_embed = self.in_fc(src)
        enc_out, (h_enc, c_enc) = self.encoder(enc_embed)
        h_enc = self._merge_bidirectional(h_enc, self.encoder.num_layers)
        c_enc = self._merge_bidirectional(c_enc, self.encoder.num_layers)
        proj_enc = torch.tanh(self.attn_fc(enc_out))

        prev = trg[:, 0:1, :]
        preds = []
        for _ in range(num_steps):
            inp_embed = self.in_fc(prev)
            attn_out, _ = self.attn(inp_embed, proj_enc, proj_enc)
            dec_input = inp_embed + attn_out
            out, (h_enc, c_enc) = self.decoder(dec_input, (h_enc, c_enc))
            raw = self.out_fc(out)
            pred_bbox = prev[:, :, :4] + raw[:, :, :4]
            pred_conf = torch.sigmoid(raw[:, :, 4:5])
            pred = torch.cat([pred_bbox, pred_conf], dim=-1)
            preds.append(pred)
            # Next input: 13-dim with predicted bbox (0:4), conf (12); zero motion (4:12)
            next_in = prev.clone()
            next_in[:, :, :4] = pred[:, :, :4]
            next_in[:, :, 12:13] = pred[:, :, 4:5]
            next_in[:, :, 4:12] = 0
            prev = next_in
        return torch.cat(preds, dim=1)

    def train_one_epoch(self, dataloader, optimizer, criterion, device="cuda"):
        self.train()
        total_loss = 0
        for batch in dataloader:
            src, trg, gt_src, gt_trg = batch
            src = src.to(device)
            trg = trg.to(device)
            gt_src = gt_src.to(device)
            gt_trg = gt_trg.to(device)

            optimizer.zero_grad()
            output = self.forward(src, trg[:, :-1])
            loss = criterion(output, gt_trg[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion, device="cuda"):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                src, trg, gt_src, gt_trg = batch
                src = src.to(device)
                trg = trg.to(device)
                gt_src = gt_src.to(device)
                gt_trg = gt_trg.to(device)
                output = self.forward(src, trg[:, :-1], teacher_forcing_ratio=0.0)
                loss = criterion(output, gt_trg[:, 1:])
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def save_weight(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path):
        self.load_state_dict(
            torch.load(path, map_location="cuda", weights_only=True)
        )
