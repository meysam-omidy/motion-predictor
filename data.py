import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import configparser

class MOTSequenceDataset(Dataset):
    def __init__(self, root_dirs, seq_in_len=20, seq_out_len=10, seq_total_len=20, transform=None):
        """
        root_dirs: list of dataset root paths, e.g., ['dancetrack/train', 'mot17/train']
        seq_len: sequence length (e.g., 10 or 20)
        """
        self.transform = transform
        self.in_sequences = []  # will hold all sequences across datasets
        self.out_sequences = []

        for root in root_dirs:
            sequences = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for seq_path in sequences:
                gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
                if not os.path.exists(gt_path):
                    continue
                cfp = configparser.ConfigParser()
                cfp.read(os.path.join(seq_path, 'seqinfo.ini'))
                df = pd.read_csv(gt_path, header=None)
                df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']

                # group by object id
                for obj_id, obj_df in df.groupby('id'):
                    obj_df = obj_df.sort_values('frame')
                    obj_df['x'] /= np.array(cfp['Sequence']['imWidth']).astype(float)
                    obj_df['y'] /= np.array(cfp['Sequence']['imHeight']).astype(float)
                    obj_df['w'] /=np.array(cfp['Sequence']['imWidth']).astype(float)
                    obj_df['h'] /= np.array(cfp['Sequence']['imHeight']).astype(float)
                    bboxes = obj_df[['x', 'y', 'w', 'h']].to_numpy()
                    # create sequences of length seq_len
                    for i in range(len(bboxes) - seq_total_len):
                        seq = bboxes[i:i+seq_total_len]
                        self.in_sequences.append(seq[:seq_in_len])
                        self.out_sequences.append(seq[-seq_out_len:])

        self.in_sequences = np.array(self.in_sequences, dtype=np.float64)
        self.out_sequences = np.array(self.out_sequences, dtype=np.float64)

    def __len__(self):
        return len(self.in_sequences)

    def __getitem__(self, idx):
        in_seq = self.in_sequences[idx]
        out_seq = self.out_sequences[idx]
        return torch.tensor(in_seq, torch.tensor(out_seq))
    
# -----------------------------
# Create train and val loaders
# -----------------------------
seq_in_len = 20
seq_out_len = 10
seq_total_len = 20
batch_size = 8192

base_dir = './'

train_dataset = MOTSequenceDataset([
    f'{base_dir}DanceTrack/train',
    f'{base_dir}MOT17/train',
    f'{base_dir}MOT20/train'
], seq_in_len=seq_in_len, seq_out_len=seq_out_len, seq_total_len=seq_total_len)

val_dataset = MOTSequenceDataset([
    f'{base_dir}DanceTrack/val',
    f'{base_dir}MOT17/val',
    f'{base_dir}MOT20/val'
], seq_in_len=seq_in_len, seq_out_len=seq_out_len, seq_total_len=seq_total_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')