import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MOTSequenceDataset(Dataset):
    def __init__(self, root_dirs, seq_len=10, transform=None):
        """
        root_dirs: list of dataset root paths, e.g., ['dancetrack/train', 'mot17/train']
        seq_len: sequence length (e.g., 10 or 20)
        """
        self.seq_len = seq_len
        self.transform = transform
        self.sequences = []  # will hold all sequences across datasets

        for root in root_dirs:
            sequences = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for seq_path in sequences:
                gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
                if not os.path.exists(gt_path):
                    continue
                df = pd.read_csv(gt_path, header=None)
                df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']

                # group by object id
                for obj_id, obj_df in df.groupby('id'):
                    obj_df = obj_df.sort_values('frame')
                    bboxes = obj_df[['x', 'y', 'w', 'h']].to_numpy()
                    # create sequences of length seq_len
                    for i in range(len(bboxes) - seq_len):
                        seq = bboxes[i:i+seq_len]
                        self.sequences.append(seq)

        self.sequences = np.array(self.sequences, dtype=np.float32)

        # optional normalization to [0,1] by assuming max image size (could parse from seqinfo.ini)
        self.sequences /= 1920.0  # example if width/height ~1920, adjust if needed

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # shape: seq_len x 4
        src = seq[:-1]  # first N-1 for input
        trg = seq[1:]   # next N-1 for target (offset prediction)
        return torch.tensor(src), torch.tensor(trg)

# -----------------------------
# Create train and val loaders
# -----------------------------
seq_len = 10
batch_size = 64

train_dataset = MOTSequenceDataset([
    'dancetrack/train', 
    'mot17/train', 
    'mot20/train'
], seq_len=seq_len)

val_dataset = MOTSequenceDataset([
    'dancetrack/val', 
    'mot17/val', 
    'mot20/val'
], seq_len=seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')