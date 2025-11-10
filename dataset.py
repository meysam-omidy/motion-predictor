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
    def compute_motion_features(bboxes):
        """
        Compute velocity and acceleration features for better motion modeling.
        Returns enhanced features: [x, y, w, h, vx, vy, vw, vh, ax, ay, aw, ah]
        
        Note: First frame has zero velocity, first two frames have zero acceleration.
        This is correct as we don't have previous frames to compute derivatives.
        """
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
    def load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, random_jump, noise_prob, noise_coeff, random_drop_prob, use_motion_features=True):
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
                
                # Compute motion features if enabled
                if use_motion_features:
                    seq_enhanced = GTSequenceDataset.compute_motion_features(seq)
                    feature_dim = 13  # 12 motion features + 1 IOU/confidence
                else:
                    seq_enhanced = np.zeros((len(seq), 5))
                    seq_enhanced[:, :4] = seq
                    feature_dim = 5
                
                seq_c = np.zeros(shape=(len(seq), feature_dim))
                seq_c_noised = np.zeros(shape=(len(seq), feature_dim))
                
                if noise_prob is not None and noise_prob > 0:
                    # Apply noise to bbox coordinates
                    seq_noised = copy(seq)
                    
                    for t in range(len(seq)):
                        if np.random.random() < noise_prob:
                            # Original bbox
                            x1, y1, x2, y2 = seq[t]
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            w, h = x2 - x1, y2 - y1
                            
                            # Add noise to center position (Gaussian)
                            # Noise proportional to bbox size
                            pos_noise_x = np.random.randn() * w * noise_coeff
                            pos_noise_y = np.random.randn() * h * noise_coeff
                            cx_noised = cx + pos_noise_x
                            cy_noised = cy + pos_noise_y
                            
                            # Add noise to scale (log-normal for positive scale changes)
                            scale_noise = np.random.randn() * noise_coeff
                            scale_factor = np.exp(scale_noise)  # Always positive
                            w_noised = w * scale_factor
                            h_noised = h * scale_factor
                            
                            # Reconstruct bbox
                            x1_noised = cx_noised - w_noised / 2
                            y1_noised = cy_noised - h_noised / 2
                            x2_noised = cx_noised + w_noised / 2
                            y2_noised = cy_noised + h_noised / 2
                            
                            # Clip to valid range [0, 1] and ensure x1 < x2, y1 < y2
                            x1_noised = np.clip(x1_noised, 0, 1)
                            y1_noised = np.clip(y1_noised, 0, 1)
                            x2_noised = np.clip(x2_noised, 0, 1)
                            y2_noised = np.clip(y2_noised, 0, 1)
                            
                            # Ensure valid bbox (min < max)
                            if x2_noised > x1_noised and y2_noised > y1_noised:
                                seq_noised[t] = [x1_noised, y1_noised, x2_noised, y2_noised]
                            # else: keep original bbox if noise makes it invalid
                    
                    # Compute IoU between original and noised sequences
                    ious = np.diag(batch_iou(seq, seq_noised))
                    
                    if use_motion_features:
                        seq_noised_enhanced = GTSequenceDataset.compute_motion_features(seq_noised)
                        seq_c_noised[:, :-1] = seq_noised_enhanced
                    else:
                        seq_c_noised[:, :4] = seq_noised
                else:
                    seq_noised = copy(seq)
                    ious = np.ones(shape=(len(seq)))
                    if use_motion_features:
                        seq_c_noised[:, :-1] = seq_enhanced
                    else:
                        seq_c_noised[:, :4] = seq_noised
                        
                seq_c_noised[:, -1] = ious
                
                if use_motion_features:
                    seq_c[:, :-1] = seq_enhanced
                else:
                    seq_c[:, :4] = seq
                seq_c[:, -1] = 1
                
                frames = frames_total[i:i+seq_total_len]

                if random_drop_prob is not None:
                    # Randomly drop frames to simulate missing detections/occlusions
                    # This makes the model robust to missing observations
                    drop_mask = np.random.random(size=len(seq_c_noised)) < random_drop_prob
                    
                    # Set confidence to 0 for dropped frames (simulating missed detection)
                    seq_c_noised[drop_mask, -1] = 0
                    
                    # Optionally zero out bbox coordinates for dropped frames
                    # This simulates complete detection failure
                    seq_c_noised[drop_mask, :4] = 0
                    
                    # If using motion features, also zero them out
                    if use_motion_features and seq_c_noised.shape[1] == 13:
                        seq_c_noised[drop_mask, 4:12] = 0
                    
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
    def from_sequence(cls, seq_path, seq_in_len=20, seq_out_len=10, seq_total_len=20, random_jump=False, noise_prob=None, noise_coeff=None, random_drop_prob=None, use_motion_features=True):
        obj = cls()
        sources, targets, gt_sources, gt_targets, (image_width, image_height) = cls.load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, random_jump, noise_prob, noise_coeff, random_drop_prob, use_motion_features)
        obj.sources = np.array(sources, dtype=np.float32)
        obj.targets = np.array(targets, dtype=np.float32)
        obj.gt_sources = np.array(gt_sources, dtype=np.float32)
        obj.gt_targets = np.array(gt_targets, dtype=np.float32)
        obj.image_width = image_width
        obj.image_height = image_height
        return obj
    

    @classmethod
    def from_roots(cls, root_dirs, seq_in_len=20, seq_out_len=10, seq_total_len=20, random_jump=False, noise_prob=None, noise_coeff=None, random_drop_prob=None, use_motion_features=True):
        sources = []
        targets = []
        gt_sources = []
        gt_targets = []


        for root in root_dirs:
            sequences = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            for seq_path in sequences:
                sources_, targets_, gt_sources_, gt_targets_, _ = cls.load_sequence(seq_path, seq_in_len, seq_out_len, seq_total_len, random_jump, noise_prob, noise_coeff, random_drop_prob, use_motion_features)
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