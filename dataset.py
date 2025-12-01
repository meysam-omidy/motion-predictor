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

def xywh_to_tlbr(bbox:np.ndarray) -> np.ndarray:
    o = np.zeros_like(bbox, dtype=float)
    o[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    o[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    o[..., 2] = bbox[..., 0] + bbox[..., 2] / 2
    o[..., 3] = bbox[..., 1] + bbox[..., 3] / 2
    return o

def batch_iou(bbox1, bbox2):
    bb1 = xywh_to_tlbr(bbox1)
    bb2 = xywh_to_tlbr(bbox2)
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
    def compute_motion_features(bboxes):
        """
        Compute velocity and acceleration features for better motion modeling.
        Returns enhanced features: [x, y, w, h, vx, vy, vw, vh, ax, ay, aw, ah]
        
        Note: First frame has zero velocity, first two frames have zero acceleration.
        This is correct as we don't have previous frames to compute derivatives.
        """
        bboxes = copy(bboxes)
        n = len(bboxes)
        enhanced = np.zeros((n, 12))
        enhanced[:, :4] = bboxes
        
        # Velocity (first-order difference)
        if n > 1:
            velocity = np.diff(bboxes, axis=0)
            enhanced[1:, 4:8] = velocity
            # First frame velocity = 0 (no previous frame)
            
        # Acceleration (second-order difference)
        if n > 2:
            acceleration = np.diff(velocity, axis=0)
            enhanced[2:, 8:12] = acceleration
            # First two frames acceleration = 0 (need at least 3 frames)
            
        return enhanced

    @staticmethod
    def load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, steps, random_jump, noise_prob, noise_coeff, random_drop_prob, use_motion_features=True):
        sources = []
        gt_sources = []
        targets = []
        gt_targets = []

        gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
        if not os.path.exists(gt_path):
            print('yes', seq_path)
            return [], [], [], [], ()
        cfp = configparser.ConfigParser()
        cfp.read(os.path.join(seq_path, 'seqinfo.ini'))
        image_width = np.array(cfp['Sequence']['imWidth']).astype(float)
        image_height = np.array(cfp['Sequence']['imHeight']).astype(float)
        borders = np.array([image_width, image_height, image_width, image_height]).astype(float)

        df = pd.read_csv(gt_path, header=None)
        df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']

        for obj_id, obj_df in df.groupby('id'):
            print(seq_path, obj_id)
            obj_df = obj_df.sort_values('frame')
            obj_df['x'] += obj_df['w'] / 2
            obj_df['y'] += obj_df['h'] / 2
            # obj_df['w'] += obj_df['x']
            # obj_df['h'] += obj_df['y']

            bboxes = obj_df[['x', 'y', 'w', 'h']].to_numpy().astype(float)
            bboxes /= borders
            frames_total = obj_df['frame'].to_numpy()

            for i in range(0, len(bboxes) - seq_total_len, steps): 
                seq = copy(bboxes[i:i+seq_total_len])
                noise = np.random.randn(*(seq.shape))
                noise[:, 0:2] *= seq[:, 2:4] * noise_coeff
                noise[:, 2:4] *= seq[:, 2:4] * noise_coeff
                seq_noised = np.where(np.random.random(size=seq.shape) < noise_prob, seq + noise, seq)
                
                if use_motion_features:
                    feature_dim = 13
                    seq_enhanced = np.zeros((len(seq), feature_dim))
                    seq_enhanced[:, :12] = GTSequenceDataset.compute_motion_features(seq_noised)
                    seq_enhanced_gt = np.zeros((len(seq), feature_dim))
                    seq_enhanced_gt[:, :12] = GTSequenceDataset.compute_motion_features(seq)
                else:
                    feature_dim = 5
                    seq_enhanced = np.zeros((len(seq), feature_dim))
                    seq_enhanced[:, :4] = seq_noised
                    seq_enhanced_gt = np.zeros((len(seq), feature_dim))
                    seq_enhanced_gt[:, :4] = seq

                seq_enhanced[:, -1] = np.diag(batch_iou(seq, seq_noised))
                seq_enhanced_gt[:, -1] = 1
                
                frames = frames_total[i:i+seq_total_len]

                # if random_drop_prob is not None:
                #     # Randomly drop frames to simulate missing detections/occlusions
                #     # This makes the model robust to missing observations
                #     drop_mask = np.random.random(size=len(seq_enhanced)) < random_drop_prob
                    
                #     # Set confidence to 0 for dropped frames (simulating missed detection)
                #     seq_enhanced[drop_mask, -1] = 0
                    
                #     # Optionally zero out bbox coordinates for dropped frames
                #     # This simulates complete detection failure
                #     seq_enhanced[drop_mask, :4] = 0
                    
                #     # If using motion features, also zero them out
                #     if use_motion_features and seq_enhanced.shape[1] == 13:
                #         seq_enhanced[drop_mask, 4:12] = 0
                    
                if not random_jump:
                    if has_jump(frames[:seq_in_len]) or has_jump(frames[-seq_out_len:]):
                        continue
                    sources.append(seq_enhanced[:seq_in_len])
                    targets.append(seq_enhanced[-seq_out_len:])
                    gt_sources.append(seq_enhanced_gt[:seq_in_len])
                    gt_targets.append(seq_enhanced_gt[-seq_out_len:])
                else:
                    index_1 = random.randint(0, int(seq_total_len / 2) - seq_in_len - 1)
                    index_2 = random.randint(0, int(seq_total_len / 2) - seq_in_len - 1)
                    if has_jump(frames[index_1: index_1 + seq_in_len]) or \
                    has_jump(frames[int(seq_total_len / 2) + index_2: int(seq_total_len / 2) + index_2 + seq_out_len:]):
                        continue
                    sources.append(seq_enhanced[index_1: index_1 + seq_in_len])
                    targets.append(seq_enhanced[int(seq_total_len / 2) + index_2: int(seq_total_len / 2) + index_2 + seq_out_len:])
                    gt_sources.append(seq_enhanced_gt[index_1: index_1 + seq_in_len])
                    gt_targets.append(seq_enhanced_gt[int(seq_total_len / 2) + index_2: int(seq_total_len / 2) + index_2 + seq_out_len:])

        return sources, targets, gt_sources, gt_targets, (image_width, image_height)


    @classmethod
    def from_sequence(cls, seq_path, seq_in_len=20, seq_out_len=10, seq_total_len=20, steps=20, random_jump=False, noise_prob=0, noise_coeff=0, random_drop_prob=None, use_motion_features=True):
        obj = cls()
        sources, targets, gt_sources, gt_targets, (image_width, image_height) = cls.load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, steps, random_jump, noise_prob, noise_coeff, random_drop_prob, use_motion_features)
        obj.sources = np.array(sources, dtype=np.float32)
        obj.targets = np.array(targets, dtype=np.float32)
        obj.gt_sources = np.array(gt_sources, dtype=np.float32)
        obj.gt_targets = np.array(gt_targets, dtype=np.float32)
        obj.image_width = image_width
        obj.image_height = image_height
        return obj
    

    @classmethod
    def from_roots(cls, root_dirs, seq_in_len=20, seq_out_len=10, seq_total_len=20, steps=20, random_jump=False, noise_prob=0, noise_coeff=0, random_drop_prob=None, use_motion_features=True):
        sources = []
        targets = []
        gt_sources = []
        gt_targets = []


        for root in root_dirs:
            sequences = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for seq_path in sequences:
                sources_, targets_, gt_sources_, gt_targets_, _ = cls.load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, steps, random_jump, noise_prob, noise_coeff, random_drop_prob, use_motion_features)
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