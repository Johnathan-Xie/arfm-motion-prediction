import numpy as np
import torch

def concatenate(x):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x)
    elif isinstance(x[0], np.ndarray):
        return np.concatenate(x)
    elif isinstance(x[0], list):
        return sum(x, [])
    else:
        raise ValueError("Unable to concatenate")
    
def apply_framewise_function_batched(func, frames, processing_batch_size=None):
    """
    Merge frame dimension into batch dimension then batch again based on batch size
    Used to process batch of video frames using frame level function
    """
    if isinstance(frames, list):
        frame_lengths = [len(v) for v in frames]
        merged_frames = concatenate(frames)
    elif type(frames) in [torch.Tensor, np.ndarray]:
        input_batch_size, frames_per_sample = frames.shape[0], frames.shape[1]
        merged_frames = frames.flatten(0, 1)
    
    if processing_batch_size is not None:
        output = []
        for start in range(0, len(merged_frames), processing_batch_size):
            end = start + processing_batch_size
            batch = merged_frames[start:end]
            output.append(func(batch))
        output = concatenate(output)
    else:
        output = func(merged_frames)
    
    if isinstance(frames, list):
        frame_intervals = np.cumsum([0] + frame_lengths)
        output = [output[frame_intervals[i]:frame_intervals[i+1]] for i in range(len(frame_intervals) - 1)]
    else:
        # Model outputs list of tensors so we split individually
        if isinstance(output, list):
            output = [tensor.view((input_batch_size, frames_per_sample) + tensor.shape[1:]) for tensor in output]
        else:
            output = output.view((input_batch_size, frames_per_sample) + output.shape[1:])
    return output