import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import configparser
from copy import copy

def has_jump(seq):
    return not ((seq[-1]- seq[0] + 1) == len(seq))

def batch_iou(bb1, bb2):
    bb1 = np.expand_dims(bb1, 1)
    bb2 = np.expand_dims(bb2, 0)
    xx1 = np.maximum(bb1[..., 0], bb2[..., 0])
    yy1 = np.maximum(bb1[..., 1], bb2[..., 1])
    xx2 = np.minimum(bb1[..., 2], bb2[..., 2])
    yy2 = np.minimum(bb1[..., 3], bb2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb1[..., 2] - bb1[..., 0]) * (bb1[..., 3] - bb1[..., 1])                                      
        + (bb2[..., 2] - bb2[..., 0]) * (bb2[..., 3] - bb2[..., 1]) - wh)                                              
    return(o) 

class GTSequenceDataset(Dataset):

    @staticmethod
    def load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, random_jump, noise_prob, noise_coeff, random_drop_prob):
        sources = []
        gt_sources = []
        targets = []
        gt_targets = []

        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        if not os.path.exists(gt_path):
            return [], []
        cfp = configparser.ConfigParser()
        cfp.read(os.path.join(seq_path, 'seqinfo.ini'))
        image_width = np.array(cfp['Sequence']['imWidth']).astype(float)
        image_height = np.array(cfp['Sequence']['imHeight']).astype(float)
        borders = np.array([image_width, image_height, image_width, image_height]).astype(float)

        df = pd.read_csv(gt_path, header=None)
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']

        for obj_id, obj_df in df.groupby('id'):
            obj_df = obj_df.sort_values('frame')
            obj_df['w'] += obj_df['x']
            obj_df['h'] += obj_df['y']
            bboxes = obj_df[['x', 'y', 'w', 'h']].to_numpy().astype(float)
            bboxes /= borders
            frames_total = obj_df['frame'].to_numpy()

            for i in range(len(bboxes) - seq_total_len): 
                seq = copy(bboxes[i:i+seq_total_len])
                seq_c = np.zeros(shape=(len(seq), 5))
                seq_c_noised = np.zeros(shape=(len(seq), 5))
                if noise_prob is not None:
                    var = (seq[1:] - seq[:-1]).mean(axis=0).__abs__() 
                    noise_probs_bool = np.random.random(size=(seq.shape[0], 1)) > noise_prob
                    noise = np.random.randn(seq.shape[0], seq.shape[1]) * var * noise_coeff
                    seq_noised = np.where(noise_probs_bool, seq, seq + noise)
                    ious = np.diag(batch_iou(seq, seq_noised))
                else:
                    seq_noised = copy(seq)
                    ious = np.ones(shape=(len(seq)))
                seq_c_noised[:, :4] = seq_noised
                seq_c_noised[:, 4] = ious
                seq_c[:, :4] = seq
                seq_c[:, 4] = 1
                frames = frames_total[i:i+seq_total_len]

                if random_drop_prob is not None:
                    pass
                    
                if not random_jump:
                    if has_jump(frames[:seq_in_len]) or has_jump(frames[-seq_out_len:]):
                        continue
                    sources.append(seq_c_noised[:seq_in_len])
                    targets.append(seq_c_noised[-seq_out_len:])
                    gt_sources.append(seq_c[:seq_in_len])
                    gt_targets.append(seq_c[-seq_out_len:])
                else:
                    index_1 = random.randint(0, int(seq_total_len / 2) - seq_in_len - 1)
                    index_2 = random.randint(0, int(seq_total_len / 2) - seq_in_len - 1)
                    if has_jump(frames[index_1: index_1 + seq_in_len]) or \
                    has_jump(frames[int(seq_total_len / 2) + index_2: int(seq_total_len / 2) + index_2 + seq_out_len:]):
                        continue
                    sources.append(seq_c_noised[index_1: index_1 + seq_in_len])
                    targets.append(seq_c_noised[int(seq_total_len / 2) + index_2: int(seq_total_len / 2) + index_2 + seq_out_len:])
                    gt_sources.append(seq_c[index_1: index_1 + seq_in_len])
                    gt_targets.append(seq_c[int(seq_total_len / 2) + index_2: int(seq_total_len / 2) + index_2 + seq_out_len:])

        return sources, targets, gt_sources, gt_targets, (image_width, image_height)


    @classmethod
    def from_sequence(cls, seq_path, seq_in_len=20, seq_out_len=10, seq_total_len=20, random_jump=False, noise_prob=None, noise_coeff=None, random_drop_prob=None):
        obj = cls()
        sources, targets, gt_sources, gt_targets, (image_width, image_height) = cls.load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, random_jump, noise_prob, noise_coeff, random_drop_prob)
        obj.sources = np.array(sources, dtype=np.float32)
        obj.targets = np.array(targets, dtype=np.float32)
        obj.gt_sources = np.array(gt_sources, dtype=np.float32)
        obj.gt_targets = np.array(gt_targets, dtype=np.float32)
        obj.image_width = image_width
        obj.image_height =image_height
        return obj
    

    @classmethod
    def from_roots(cls, root_dirs, seq_in_len=20, seq_out_len=10, seq_total_len=20, random_jump=False, noise_prob=None, noise_coeff=None, random_drop_prob=None):
        sources = []
        targets = []
        gt_sources = []
        gt_targets = []


        for root in root_dirs:
            sequences = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for seq_path in sequences:
                sources_, targets_, gt_sources_, gt_targets_, _ = cls.load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, random_jump, noise_prob, noise_coeff, random_drop_prob)
                sources.extend(sources_)
                targets.extend(targets_)
                gt_sources.extend(gt_sources_)
                gt_targets.extend(gt_targets_)

        obj = cls()
        obj.sources = np.array(sources, dtype=np.float32)
        obj.targets = np.array(targets, dtype=np.float32)
        obj.gt_sources = np.array(gt_sources, dtype=np.float32)
        obj.gt_targets = np.array(gt_targets, dtype=np.float32)
        return obj

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]
        gt_source = self.gt_sources[idx]
        gt_target = self.gt_targets[idx]
        return torch.tensor(source), torch.tensor(target), torch.tensor(gt_source), torch.tensor(gt_target)