from torch import nn
import torch
import random
import os


class ImprovedLSTMPredictor(nn.Module):
    """
    Decoder-only LSTM: a single LSTM processes src then target steps.
    - Feed src through the LSTM to get (h, c).
    - Then loop over target steps: input (teacher or prev pred) → LSTM step → out_fc → predict next (5-dim).
    - input_dim=13, output_dim=5 (bbox residual + sigmoid confidence).
    """

    def __init__(
        self,
        input_dim=13,
        output_dim=5,
        d_model=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.teacher_forcing_ratio = teacher_forcing_ratio

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

        self.lstm = nn.LSTM(
            d_model,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

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

    def forward(self, src, trg, teacher_forcing_ratio=None):
        """
        src: (B, S, input_dim), trg: (B, T, input_dim).
        Returns: (B, T, output_dim) — one prediction per target step (vs gt_trg[:, 1:]).
        """
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        _, trg_len, _ = trg.size()

        # Single LSTM: run over src to get (h, c)
        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)

        # Same LSTM: loop over target steps and predict
        prev = trg[:, 0:1, :]
        outputs = []
        for t in range(trg_len):
            inp_embed = self.in_fc(prev)
            out, (h, c) = self.lstm(inp_embed, (h, c))
            raw = self.out_fc(out)
            pred_bbox = prev[:, :, :4] + raw[:, :, :4]
            pred_conf = torch.sigmoid(raw[:, :, 4:5])
            pred = torch.cat([pred_bbox, pred_conf], dim=-1)
            outputs.append(pred)
            if t + 1 < trg_len:
                use_teacher = random.random() < teacher_forcing_ratio
                if use_teacher:
                    prev = trg[:, t + 1 : t + 2, :]
                else:
                    next_in = prev.clone()
                    next_in[:, :, :4] = pred[:, :, :4]
                    next_in[:, :, 12:13] = pred[:, :, 4:5]
                    next_in[:, :, 4:12] = 0
                    prev = next_in
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def inference(self, src, trg, num_steps=1):
        """Autoregressive inference. trg has at least first step (B, 1, 13). Returns (B, num_steps, 5)."""
        src_embed = self.in_fc(src)
        _, (h, c) = self.lstm(src_embed)

        prev = trg[:, 0:1, :]
        preds = []
        for _ in range(num_steps):
            inp_embed = self.in_fc(prev)
            out, (h, c) = self.lstm(inp_embed, (h, c))
            raw = self.out_fc(out)
            pred_bbox = prev[:, :, :4] + raw[:, :, :4]
            pred_conf = torch.sigmoid(raw[:, :, 4:5])
            pred = torch.cat([pred_bbox, pred_conf], dim=-1)
            preds.append(pred)
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
