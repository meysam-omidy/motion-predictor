import torch
from torch import nn
from copy import copy

class LSTM(nn.Module):
    def __init__(self, input_size, middle_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.middle_size = middle_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc_in = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size, middle_size),
            nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(middle_size, hidden_size),
            # nn.LeakyReLU()
        )
        self.lstm = nn.LSTM(middle_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, middle_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size, middle_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size, input_size),
            nn.Sigmoid(),
        )
        # self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, input_h=None, input_c=None):
        if input_h == None:
            h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).type(torch.float).to(x.device)
        else:
            h = input_h
        if input_c == None:
            c = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).type(torch.float).to(x.device)
        else:
            c = input_c
        o = self.fc_in(x)
        o, (h, c) = self.lstm(o, (h, c))
        o = self.fc_out(o)
        return o * 2 - 1, (h, c)

    def sequential_forward(self, x, teacher_ratio):
        h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).type(torch.float).to(x.device)
        c = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).type(torch.float).to(x.device)
        outputs = torch.zeros_like(x)
        last_output = x[:, 0, :].unsqueeze(1)
        for t in range(x.shape[1]):
            mask = torch.rand_like(x[:, t, :].unsqueeze(1), dtype=torch.float).to(x.device)
            input_tensor = torch.where(mask < teacher_ratio, x[:, t, :].unsqueeze(1), last_output)
            o, (h,c) = self.forward(input_tensor, h, c)
            # o, (h, c) = self.lstm(input_tensor, (h, c))
            # o = self.fc(o)
            outputs[:, t, :] = o.squeeze()
            last_output = o
        return outputs * 2 - 1

    def sequentia_inference(self, x, input_h=None, input_c=None, num_steps=1):
        if input_h == None:
            h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).type(torch.float).to(x.device)
        else:
            h = input_h
        if input_c == None:
            c = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).type(torch.float).to(x.device)
        else:
            c = input_c
        outputs = torch.zeros(size=(x.shape[0], num_steps, self.input_size)).to(x.device)
        last_output = x[:, 0, :].unsqueeze(1)
        for t in range(num_steps):
            # mask = torch.rand_like(x[:, t, :].unsqueeze(1), dtype=torch.float).to(x.device)
            # input_tensor = torch.where(mask < teacher_ratio, x[:, t, :].unsqueeze(1), last_output)
            o, (h,c) = self.forward(last_output, h, c)
            # o, (h, c) = self.lstm(input_tensor, (h, c))
            # o = self.fc(o)
            outputs[:, t, :] = o.squeeze()
            last_output = o
        return outputs * 2 - 1

class SequenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        diff = pred - target
        l1 = torch.abs(diff)[mask].mean()
        l2 = (diff ** 2)[mask].mean()
        return l1 + l2


# Hyperparameters
EPOCHS = 150
initial_teacher_ratio = 1
lst_model = LSTM(4, 32, 64, 8, dropout=0.1).cuda()
criterion = SequenceLoss()
optimizer = torch.optim.Adam(lst_model.parameters(), lr=5e-2, weight_decay=1e-5)
# Scheduler after EPOCHS defined
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)