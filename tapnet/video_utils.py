import numpy as np

def get_bounds(detected_similar, total_len):
    if detected_similar.shape[0] == 0:
        return 0, total_len
    idx = 0
    while idx < len(detected_similar) and (detected_similar[idx] == idx):
        idx += 1
    low_crop = idx
    idx = 0
    while idx < len(detected_similar) and (detected_similar[len(detected_similar) - idx - 1].squeeze() == total_len - idx - 1):
        idx += 1
    high_crop = total_len - idx
    return low_crop, high_crop


def crop_still_edges(frames, similarity_threshold=20.0, shift=0):
    if shift > 0:
        frame_difference_height = (frames[0, :, :-shift].astype(float) - frames[-1, :, shift:].astype(float))
        frame_difference_width = (frames[0, :-shift].astype(float) - frames[-1, shift:].astype(float))
    else:
        frame_difference_height = (frames[0].astype(float) - frames[-1].astype(float))
        frame_difference_width = (frames[0].astype(float) - frames[-1].astype(float))

    height_diffs = frame_difference_height.max(axis=(1, 2))

    height_detected_similar = np.argwhere(height_diffs < similarity_threshold)
    top_crop, bottom_crop = get_bounds(height_detected_similar, frames.shape[1])

    width_diffs = frame_difference_width.max(axis=(0, 2))

    width_detected_similar = np.argwhere(width_diffs < similarity_threshold)
    left_crop, right_crop = get_bounds(width_detected_similar, frames.shape[2])
    
    return frames[:, top_crop:bottom_crop, left_crop:right_crop]

def temporal_difference(frames):
    return (frames[:-1].astype(float) - frames[1:].astype(float)) ** 2

def camera_motion_detection(frames, edge_ratio=1 / 16, mse_threshold=30.0, total_ratio=0.1):
    height_edge_size = round(frames.shape[1] * edge_ratio)
    width_edge_size = round(frames.shape[1] * edge_ratio)
    
    edges = [frames[:, -height_edge_size:], frames[:, :, :width_edge_size], frames[:, :, -width_edge_size:]]
    diffs = [temporal_difference(edge).mean((1, 2, 3)) for edge in edges]
    
    likely_motion = np.stack([d > mse_threshold for d in diffs])
    likely_motion = np.logical_and.reduce(likely_motion, axis=0)
    return likely_motion.mean() > total_ratio