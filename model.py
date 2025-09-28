import torch
import torch.nn as nn
import torch.optim as optim
import random

# ----------------------------
# LSTM Seq2Seq for Bounding Box Prediction

# ----------------------------

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float64)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float64)
        self.fc_out = nn.Linear(hidden_dim, input_dim, dtype=torch.float64)  # predict offset (dx, dy, dw, dh)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        # src: (batch, seq_len, 4)
        # trg: (batch, seq_len, 4) - ground truth future sequence

        batch_size, seq_len, _ = src.size()
        outputs = []

        # Encode
        _, (hidden, cell) = self.encoder(src)

        # First input to decoder is the last frame of src
        decoder_input = src[:, -1:, :]  # shape (batch, 1, 4)

        for t in range(seq_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc_out(out)  # (batch, 1, 4)
            outputs.append(pred)

            # Decide if we use teacher forcing
            use_teacher = trg is not None and random.random() < teacher_forcing_ratio
            decoder_input = trg[:, t:t+1, :] if use_teacher else pred

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, 4)
        return outputs

# ----------------------------
# Training & Evaluation Loops
# ----------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)  # no teacher forcing at eval
            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# ----------------------------
# Main Training Script
# ----------------------------

# def train_model(train_loader, val_loader, num_epochs=20, lr=1e-3, teacher_forcing_ratio=0.5, device="cuda"):
model = LSTMPredictor().to(device)
criterion = nn.MSELoss()  # predicting offsets → regression
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

num_epochs = 50
lr = 1e-3
teacher_forcing_ratio = 1   ``
device="cuda"

best_val_loss = float("inf")

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio)
    val_loss = evaluate(model, val_loader, criterion, device)

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_lstm_model.pth")

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}: Train Loss = {train_loss:.8f}, Val Loss = {val_loss:.8f}, LR = {current_lr:.8f}")

print("Training complete. Best Val Loss:", best_val_loss)
return model


# train_model(train_loader, val_loader, num_epochs=50, lr=1e-3, teacher_forcing_ratio=1)
