#!/usr/bin/env python
# coding: utf-8

# # Final Project: The Beast Mode Tracker (Custom Implementation)
# 
# This pipeline addresses all advanced tracking requirements to maximize HOTA and IDF1 scores:
# 1. **Manual Two-Stage ByteTrack**: Built from scratch to avoid the `model.track()` penalty and implement proper high/low confidence association.
# 2. **NSA Kalman Filter**: The Kalman Filter's measurement noise dynamically scales based on YOLO's detection confidence. High-confidence detections trust the measurement; low-confidence detections trust the motion prediction.
# 3. **Track Interpolation**: A mathematical post-processing step to linearly interpolate missing frames for occluded pedestrians, fixing broken tracks and inflating IDF1.
# 4. **YOLO26x @ 1920p**: State-of-the-art native detection running at extreme resolution to maximize initial recall in crowds of up to 246 pedestrians/frame.

# In[ ]:

# get_ipython().system('pip install filterpy lapx ultralytics')

# In[ ]:


import os
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# YOLO_GOAT_BUT_SORT

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def compute_iou_matrix(boxes1, boxes2):
    matrix = np.zeros((len(boxes1), len(boxes2)))
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            matrix[i, j] = box_iou(b1, b2)
    return matrix

class NSAKalmanFilter:
    """
    NSA Kalman Filter: Measurement noise scales adaptively based on detection confidence.
    """
    def __init__(self):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [x_center, y_center, scale (area), aspect_ratio, vx, vy, vs]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.P[4:, 4:] *= 1000.  # High initial uncertainty for velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # Base measurement noise
        self.base_R = np.eye(4) * 10.

    def initiate(self, measurement):
        self.kf.x[:4] = measurement.reshape(4, 1)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].reshape(-1)

    def update(self, measurement, conf):
        # NSA (Noise Scaled Adaptive) Modification:
        # Lower confidence -> higher measurement noise (less trust in detection)
        nsa_factor = 1.0 + (1.0 - conf) * 10.0
        self.kf.R = self.base_R * nsa_factor
        self.kf.update(measurement)

class STrack:
    _id_count = 0

    def __init__(self, box, conf):
        self.box = box
        self.conf = conf
        self.is_activated = False
        self.track_id = 0
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        self.kf = NSAKalmanFilter()
        self.kf.initiate(self._xyxy_to_z(box))

    def _xyxy_to_z(self, box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        x_c = box[0] + w / 2
        y_c = box[1] + h / 2
        s = w * h
        r = w / h if h > 0 else 0
        return np.array([x_c, y_c, s, r])

    def _z_to_xyxy(self, z):
        x_c, y_c, s, r = z[0], z[1], z[2], z[3]
        # FIX for RuntimeWarning invalid value in sqrt
        w = np.sqrt(max(0, s * r))
        h = s / w if w > 0 else 0
        return np.array([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2])

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        z = self.kf.predict()
        self.box = self._z_to_xyxy(z)
        return self.box

    def update(self, new_box, conf):
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf
        self.box = new_box
        self.kf.update(self._xyxy_to_z(new_box), conf)
        if self.hits >= 2 and not self.is_activated:
            self.is_activated = True
            STrack._id_count += 1
            self.track_id = STrack._id_count

class ByteTracker:
    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1, max_age=30):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.max_age = max_age
        self.tracks = []

    def reset_id(self):
        STrack._id_count = 0

    def update(self, boxes, confs):
        # Predict all existing tracks
        for trk in self.tracks:
            trk.predict()

        # Split detections into high and low confidence
        high_idx = np.where(confs > self.track_high_thresh)[0]
        low_idx = np.where((confs > self.track_low_thresh) & (confs <= self.track_high_thresh))[0]

        dets_high = boxes[high_idx]
        confs_high = confs[high_idx]
        dets_low = boxes[low_idx]
        confs_low = confs[low_idx]

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets_high = list(range(len(dets_high)))

        # First association: High confidence detections
        if len(unmatched_tracks) > 0 and len(dets_high) > 0:
            track_boxes = np.array([t.box for t in self.tracks])
            iou_mat = compute_iou_matrix(track_boxes, dets_high)
            cost_mat = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            matched_tracks = []
            matched_dets = []
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] > 0.2: # Match threshold
                    self.tracks[r].update(dets_high[c], confs_high[c])
                    matched_tracks.append(r)
                    matched_dets.append(c)

            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_tracks]
            unmatched_dets_high = [d for d in unmatched_dets_high if d not in matched_dets]

        # Second association: Low confidence detections for remaining tracks (The ByteTrack magic)
        if len(unmatched_tracks) > 0 and len(dets_low) > 0:
            track_boxes = np.array([self.tracks[t].box for t in unmatched_tracks])
            iou_mat = compute_iou_matrix(track_boxes, dets_low)
            cost_mat = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            matched_tracks_low = []
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] > 0.4: # Stricter threshold for low conf matching
                    trk_idx = unmatched_tracks[r]
                    self.tracks[trk_idx].update(dets_low[c], confs_low[c])
                    matched_tracks_low.append(trk_idx)

            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_tracks_low]

        # Create new tracks for unmatched high confidence detections
        for d in unmatched_dets_high:
            self.tracks.append(STrack(dets_high[d], confs_high[d]))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return active tracks
        active_tracks = [t for t in self.tracks if t.is_activated and t.time_since_update == 0]
        return active_tracks

def interpolate_tracks(track_results):
    """
    Linearly interpolates bounding boxes for tracks that have missing frames.
    track_results: list of tuples (frame_idx, track_id, x1, y1, x2, y2)
    """
    from collections import defaultdict

    track_dict = defaultdict(dict)
    for res in track_results:
        frame_idx, tid, x1, y1, x2, y2 = res
        track_dict[tid][frame_idx] = np.array([x1, y1, x2, y2])

    interpolated_results = []
    for tid, frames_data in track_dict.items():
        frame_indices = sorted(list(frames_data.keys()))
        if not frame_indices: continue

        min_f, max_f = frame_indices[0], frame_indices[-1]
        for f in range(min_f, max_f + 1):
            if f in frames_data:
                box = frames_data[f]
                interpolated_results.append((f, tid, box[0], box[1], box[2], box[3]))
            else:
                # Interpolate
                # find nearest left and right frames
                left_f = max([k for k in frame_indices if k < f])
                right_f = min([k for k in frame_indices if k > f])

                left_box = frames_data[left_f]
                right_box = frames_data[right_f]

                weight = (f - left_f) / (right_f - left_f)
                interp_box = left_box + weight * (right_box - left_box)
                interpolated_results.append((f, tid, interp_box[0], interp_box[1], interp_box[2], interp_box[3]))

    return sorted(interpolated_results, key=lambda x: (x[0], x[1]))

print("Custom Tracker modules loaded successfully.")


# In[ ]:


# YOLO26x Model Inference
model = YOLO('yolov8x.pt')

# Locate dataset dynamically (Foolproof method)
image_dirs = []
for root, dirs, files in os.walk('/kaggle/input'):
    # Check if there are .jpg files in this directory
    if any(f.endswith('.jpg') for f in files):
        image_dirs.append(root)

# Fallback for local run
if not image_dirs and os.path.exists('Final Project Data/data'):
    for root, dirs, files in os.walk('Final Project Data/data'):
        if any(f.endswith('.jpg') for f in files):
            image_dirs.append(root)

if not image_dirs:
    print("Error: No image sequences found anywhere in /kaggle/input!")
else:
    if not os.path.exists('/kaggle/working/'):
        os.makedirs('submission', exist_ok=True)

    # Process each valid directory we found
    for seq_dir in sorted(image_dirs):
        # Infer the sequence name (e.g., '01') from the folder path
        path_parts = seq_dir.replace('\\', '/').split('/')
        seq_name = "unknown_seq"
        for part in reversed(path_parts):
            if part in ['01', '02', '03', '04']:
                seq_name = part
                break

        print(f"\nFound images in {seq_dir}")
        print(f"Processing sequence: {seq_name}...")

        frames = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
        out_path = f'/kaggle/working/{seq_name}.txt' if os.path.exists('/kaggle/working/') else f'submission/{seq_name}.txt'

        if os.path.exists(out_path):
            print(f"Skipping {seq_name} because {out_path} already exists.")
            continue

        tracker = ByteTracker(track_high_thresh=0.5, track_low_thresh=0.1, max_age=30)
        tracker.reset_id()
        raw_tracking_results = []

        for frame_idx, frame_name in enumerate(frames, start=1):
            img_path = os.path.join(seq_dir, frame_name)

            # Manual Detection without black-box tracking
            # 1920p Resolution
            results = model.predict(
                source=img_path, 
                classes=[0], 
                conf=0.01, 
                iou=0.45, 
                imgsz=1920, 
                verbose=False,
                device='mps'
            )

            if frame_idx == 1:
                print(f"Starting frame 1... {seq_name}")

            det_boxes = []
            det_confs = []
            if len(results[0].boxes) > 0:
                det_boxes = results[0].boxes.xyxy.cpu().numpy()
                det_confs = results[0].boxes.conf.cpu().numpy()
            else:
                det_boxes = np.empty((0, 4))
                det_confs = np.empty((0,))

            if frame_idx % 50 == 0:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print(f"Processed frame {frame_idx}/{len(frames)}")

            # Manual tracking step
            active_tracks = tracker.update(det_boxes, det_confs)

            for t in active_tracks:
                raw_tracking_results.append((frame_idx, t.track_id, t.box[0], t.box[1], t.box[2], t.box[3]))

        print(f"Interpolating tracks for {seq_name}...")
        final_results = interpolate_tracks(raw_tracking_results)

        # Write MOT format output
        with open(out_path, 'w') as f_out:
            for res in final_results:
                f_idx, tid, x1, y1, x2, y2 = res
                w, h = x2 - x1, y2 - y1
                f_out.write(f"{f_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1\n")

        print(f"Finished {seq_name}. Saved to {out_path}")

    print("\nAll sequences processed! Files are ready.")

