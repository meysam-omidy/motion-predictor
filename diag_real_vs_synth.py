"""Measure the REAL detector's error/confidence distribution vs the synthetic
noise the adaptive_kalman model was trained on. Confirms train/inference mismatch."""
import os, glob
import numpy as np
import pandas as pd

DATASET = 'C:/Projects/.Datasets/DanceTrack/train'
DETS = 'C:/Projects/.Detections/DanceTrack'
IOU_THR = 0.5


def tlbr(b):  # from tlwh
    return np.array([b[0], b[1], b[0]+b[2], b[1]+b[3]])


def iou(a, B):
    # a: (4,) tlbr, B: (N,4) tlbr
    xx1 = np.maximum(a[0], B[:,0]); yy1 = np.maximum(a[1], B[:,1])
    xx2 = np.minimum(a[2], B[:,2]); yy2 = np.minimum(a[3], B[:,3])
    w = np.clip(xx2-xx1, 0, None); h = np.clip(yy2-yy1, 0, None)
    inter = w*h
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_B = (B[:,2]-B[:,0])*(B[:,3]-B[:,1])
    return inter/(area_a+area_B-inter+1e-9)


seqs = sorted(os.listdir(DATASET))[:6]
confs, err_x, err_y, err_w, err_h = [], [], [], [], []
n_gt = n_matched = 0

for seq in seqs:
    gtp = os.path.join(DATASET, seq, 'gt', 'gt.txt')
    detp = os.path.join(DETS, seq + '.txt')
    if not (os.path.exists(gtp) and os.path.exists(detp)):
        continue
    gt = pd.read_csv(gtp, header=None).to_numpy()  # frame,id,x,y,w,h,...
    det = np.loadtxt(detp, delimiter=',')           # frame,x1,y1,x2,y2,score
    for fr in np.unique(gt[:,0]):
        gts = gt[gt[:,0]==fr]
        ds = det[det[:,0]==fr]
        if len(ds)==0:
            n_gt += len(gts); continue
        dboxes = ds[:,1:5]; dscores = ds[:,5]
        for g in gts:
            n_gt += 1
            gb = g[2:6]  # tlwh
            ious = iou(tlbr(gb), dboxes)
            j = int(np.argmax(ious))
            if ious[j] < IOU_THR:
                continue
            n_matched += 1
            db = dboxes[j]  # tlbr
            # convert det tlbr -> tlwh center
            dw = db[2]-db[0]; dh = db[3]-db[1]
            dcx = db[0]+dw/2; dcy = db[1]+dh/2
            gcx = gb[0]+gb[2]/2; gcy = gb[1]+gb[3]/2
            confs.append(dscores[j])
            err_x.append((dcx-gcx)/max(gb[2],1)); err_y.append((dcy-gcy)/max(gb[3],1))
            err_w.append((dw-gb[2])/max(gb[2],1)); err_h.append((dh-gb[3])/max(gb[3],1))

confs=np.array(confs); ex=np.array(err_x); ey=np.array(err_y); ew=np.array(err_w); eh=np.array(err_h)
print(f'seqs={len(seqs)} GT boxes={n_gt} matched={n_matched} recall={n_matched/max(n_gt,1):.3f}')
print(f'REAL confidence: mean={confs.mean():.3f} std={confs.std():.3f} p10={np.percentile(confs,10):.3f} p50={np.percentile(confs,50):.3f}')
print('REAL normalized error (err/box_size), std per coord:')
print(f'  x={ex.std():.4f}  y={ey.std():.4f}  w={ew.std():.4f}  h={eh.std():.4f}')
print(f'REAL |error| means: x={np.abs(ex).mean():.4f} y={np.abs(ey).mean():.4f} w={np.abs(ew).mean():.4f} h={np.abs(eh).mean():.4f}')
# error vs confidence: split by conf median
hi = confs>=np.median(confs); lo=~hi
print(f'high-conf(>{np.median(confs):.2f}) err_w std={ew[hi].std():.4f}  low-conf err_w std={ew[lo].std():.4f}')
print(f'high-conf err_h std={eh[hi].std():.4f}  low-conf err_h std={eh[lo].std():.4f}')
# correlation conf vs |scale error|
scale_err = (np.abs(ew)+np.abs(eh))/2
print(f'corr(confidence, |scale error|) = {np.corrcoef(confs, scale_err)[0,1]:.3f}')
print()
print('SYNTHETIC (training): noise on 30%% of frames, std=0.1*box_size per coord;')
print('  so per-coord error std over all frames ~= 0.1*sqrt(0.30) ~= 0.055 (normalized); conf=IoU(GT,noised).')
