import torch

import numpy as np
from tqdm import tqdm

from transformers import BatchFeature
from typing import Mapping
from copy import deepcopy

def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
    thresholds = [1, 2, 4, 8, 16]
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts.

    Within Thresh, Occ.

    Acc.)

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.

    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
        query_points: The query points, an in the format [t, y, x].  Its size is
        [b, n, 3], where b is the batch size and n is the number of queries
        gt_occluded: A boolean array of shape [b, n, t], where t is the number of
        frames.  True indicates that the point is occluded.
        gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
        format [x, y]
        pred_occluded: A boolean array of predicted occlusions, in the same format
        as gt_occluded.
        pred_tracks: An array of track predictions from your algorithm, in the same
        format as gt_tracks.
        query_mode: Either 'first' or 'strided', depending on how queries are
        sampled.  If 'first', we assume the prior knowledge that all points
        before the query point are occluded, and these are removed from the
        evaluation.
        get_trackwise_metrics: if True, the metrics will be computed for every
        track (rather than every video, which is the default).  This means
        every output tensor will have an extra axis [batch, num_tracks] rather
        than simply (batch).

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
            predicted to be within the given pixel threshold, ignoring occlusion
            prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
            threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)

    metrics = {}

    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == 'first':
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == 'strided':
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError('Unknown query mode ' + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=summing_axis,
    ) / np.sum(evaluation_points, axis=summing_axis)
    metrics['occlusion_accuracy'] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in thresholds:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=summing_axis,
        )
        count_visible_points = np.sum(
            visible & evaluation_points, axis=summing_axis
        )
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=summing_axis
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(
            false_positives & evaluation_points, axis=summing_axis
        )
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics['average_jaccard'] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics['average_pts_within_thresh'] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics


def get_preds_labels_visibles(
    model,
    loader,
    num_pred_per_sample=5,
    sampling_kwargs={
        "use_kv_cache": True,
        "num_steps": 16
    },
    is_world_process_zero=True,
    accelerator=None,
    evaluate_forecast_only=True,
    provide_tracks_for_visible_frames=False,
):
    all_preds, all_labels, all_visibles, all_query_points = [], [], [], []
    for batch in tqdm(loader):
        keys = ["pixel_values", "global_pixel_values", "attention_mask", "input_ids", "camera_motion", "frame_rate", "text_input_ids", "text_attention_mask"]
        inference_batch = BatchFeature({k:batch.get(k) for k in batch if k in keys if batch.get(k) is not None})
        num_visible_frames = inference_batch["pixel_values"].shape[1]
        provided_track_len = num_visible_frames - 1 if provide_tracks_for_visible_frames else 0
        inference_batch["input_ids"] = inference_batch["input_ids"][:, :, :(provided_track_len + 1)]

        if accelerator is None:
            inference_batch = inference_batch.to(model.device)
        
        pred_start = inference_batch["pixel_values"].shape[1] if evaluate_forecast_only else 0
        for k in inference_batch:
            inference_batch[k] = inference_batch[k].repeat_interleave(num_pred_per_sample, dim=0)
        output = model.sample(
            **inference_batch,
            **sampling_kwargs
        )
        predicted_tracks = output.predictor_output[:, :, (pred_start + 1):]
        predicted_tracks = predicted_tracks.view((len(batch["labels"]), num_pred_per_sample) + predicted_tracks.shape[1:])

        if accelerator is not None:
            batch_preds = accelerator.gather_for_metrics(predicted_tracks).cpu()
            batch_labels = accelerator.gather_for_metrics(batch["labels"][:, :, pred_start:]).cpu()
            batch_visibles = accelerator.gather_for_metrics(batch["label_mask"][:, :, pred_start:]).cpu()
            batch_query_points = accelerator.gather_for_metrics(batch["input_ids"][:, :, 0]).cpu()
            all_preds.append(batch_preds)
            all_labels.append(batch_labels)
            all_visibles.append(batch_visibles)
            all_query_points.append(batch_query_points)
        else:
            all_preds.append(predicted_tracks.cpu())
            all_query_points.append(batch["input_ids"][:, :, 0].cpu())
            all_labels.append(batch["labels"][:, :, pred_start:].cpu())
            all_visibles.append(batch["label_mask"][:, :, pred_start:].cpu())
    model.reset_caches()
    if is_world_process_zero:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_visibles = torch.cat(all_visibles).numpy()
        all_query_points = torch.cat(all_query_points).numpy()
        return all_preds, all_labels, all_visibles, all_query_points
    else:
        return None, None, None, None

import torch
import numpy as np
# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def mmd(x, y):
    """
    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
        The MMD distance between x and y embedding sets.
    """
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)

def get_angle_magnitude_features(values):
    if values.shape[-1] == 2:
        angle_features = np.arctan2(values[..., 0], values[..., 1])[..., None]
    else:
        angle_features_x = np.arctan2(values[..., 1], values[..., 2])
        angle_features_y = np.arctan2(values[..., 0], values[..., 2])
        angle_features_z = np.arctan2(values[..., 1], values[..., 2])
        angle_features = np.concatenate((angle_features_x, angle_features_y, angle_features_z), axis=-1)[..., None]
    
    magnitude_features = np.log(np.linalg.norm(values, axis=-1) + 1)[..., None]
    return np.concatenate((angle_features, magnitude_features), axis=-1)

def get_per_track_motion_features(tracks):
    velocities = np.concatenate((np.zeros_like(tracks[:, :, :1]), tracks[:, :, 1:] - tracks[:, :, :-1]), axis=2)
    accelerations = np.concatenate((np.zeros_like(velocities[:, :, :1]), velocities[:, :, 1:] - velocities[:, :, :-1]), axis=2)

    velocity_angle_magnitude_features = get_angle_magnitude_features(velocities)
    acceleration_angle_magnitude_features = get_angle_magnitude_features(accelerations)
    
    return np.concatenate((velocity_angle_magnitude_features, acceleration_angle_magnitude_features), axis=-1)

def get_joint_features(tracks, temporal_window=16, temporal_window_stride=16):
    per_track_features = get_per_track_motion_features(tracks)
    temporal_window_features = torch.from_numpy(per_track_features).unfold(2, temporal_window, temporal_window_stride).flatten(-2, -1).numpy()
    return temporal_window_features

def track_feature_mmd(track_set_1, track_set_2, **feature_kwargs):
    motion_feature_set_1 = get_joint_features(track_set_1, **feature_kwargs)
    motion_feature_set_2 = get_joint_features(track_set_2, **feature_kwargs)
    mmd_value = mmd(motion_feature_set_1.reshape(-1, motion_feature_set_1.shape[-1]), motion_feature_set_2.reshape(-1, motion_feature_set_2.shape[-1]))

    return mmd_value

def calculate_metrics(
    all_preds,
    all_labels,
    all_visibles,
    all_query_points,
    metric_for_best="average_pts_within_thresh",
    thresholds=[1, 2, 4, 8, 16]
):
    all_metrics = dict()
    if all_query_points.shape[-1] != 3:
        all_query_points = np.concatenate((np.zeros(all_query_points.shape[:-1] + (1,)), all_query_points), axis=-1)
    for data_idx in range(all_preds.shape[0]):
        # point tracker metrics
        data_metrics = {}
        mask = all_visibles[data_idx].sum(-1) > 0
        data_preds = all_preds[data_idx, :, mask].swapaxes(0, 1)
        labels = all_labels[None, data_idx, mask]
        visibles = all_visibles[None, data_idx, mask]
        query_points = all_query_points[None, data_idx, mask]

        for sample_idx in range(all_preds.shape[1]):
            preds = data_preds[None, sample_idx]
            metrics = compute_tapvid_metrics(
                query_points=query_points,
                gt_occluded=~visibles,
                gt_tracks=labels,
                pred_occluded=~visibles,
                pred_tracks=preds,
                query_mode="first",
                thresholds=thresholds
            )
            if len(data_metrics) == 0 or data_metrics[metric_for_best] < metrics[metric_for_best]:
                data_metrics = metrics
            
        for k,v in data_metrics.items():
            if all_metrics.get(k) is None:
                all_metrics[k] = [v,]
            else:
                all_metrics[k].append(v)
        
    for k,v in all_metrics.items():
        all_metrics[k] = sum(v) / len(v)
    # motion feature metrics
    # mask for non visible tracks?
    # 0th prediction selected
    print(all_preds[:, 0].shape, all_labels.shape)
    all_metrics["motion_feature_mmd"] = track_feature_mmd(all_preds[:, 0], all_labels)
    return all_metrics

class EvaluationProcessorTransformWrapper:
    def __init__(
        self,
        processor,
        eval_max_frames=8,
        total_points_needed_before_selection=None,
        movement_weighting_temperature=0.0,
        visible_min_ratio=0.0,
        mask_camera_motion_indicator=True,
        track_skip=1,
        also_skip_video=True
    ):
        self.processor = processor
        self.eval_max_frames = eval_max_frames
        self.total_points_needed_before_selection = total_points_needed_before_selection
        self.movement_weighting_temperature = movement_weighting_temperature
        self.visible_min_ratio = visible_min_ratio
        self.mask_camera_motion_indicator = mask_camera_motion_indicator
        self.track_skip = track_skip
        self.also_skip_video = also_skip_video

    def __call__(self, examples, return_visualization_info=False, start_indices=None):
        if isinstance(examples, dict):
            videos = examples["video"]
            tracks = examples["tracks"]
            query_points = examples["query_points"]
            visibles = examples["visibles"]
            camera_motion = examples.get("camera_motion")
            text_inputs = examples.get("text")
        elif isinstance(examples, list):
            videos = [i["video"] for i in examples]
            tracks = [i["tracks"] for i in examples]
            query_points = [i["query_points"] for i in examples]
            visibles = [i["visibles"] for i in examples]
            camera_motion = [i.get("camera_motion") for i in examples]
            text_inputs = [i.get("text") for i in examples]
        
        if self.mask_camera_motion_indicator and camera_motion is not None:
            camera_motion = [None for _ in range(len(camera_motion))]
        
        if self.track_skip > 1:
            tracks = [np.array(i) for i in tracks]
            tracks = [tracks[i][:, ::self.track_skip] for i in range(len(tracks))]
            
            visibles = [np.array(i) for i in visibles]
            visibles = [visibles[i][:, ::self.track_skip] for i in range(len(visibles))]
            
            if self.also_skip_video:
                for i in range(len(videos)):
                    videos[i]["frames"] = videos[i]["frames"][::self.track_skip]
        
        if start_indices is None:
            start_indices = [0] * len(videos)
        
        if self.track_skip > 1 and not self.also_skip_video:
            video_start_indices = deepcopy(start_indices)
            video_start_indices = [video_start_indices[i] * self.track_skip if not self.also_skip_video else video_start_indices[i] for i in range(len(videos))]
        else:
            video_start_indices = None
        original_num_query_points = [len(i) for i in query_points]
        
        processor_output = self.processor(
            videos=videos,
            tracks=tracks,
            query_points=query_points,
            visibles=visibles,
            max_frames=self.eval_max_frames,
            start_indices=start_indices,
            video_start_indices=video_start_indices,
            camera_motion=camera_motion,
            text_inputs=text_inputs,
            movement_weighting_temperature=self.movement_weighting_temperature,
            mask_non_visible_tracks=True,
            visible_min_ratio=self.visible_min_ratio,
            total_points_needed_before_selection=self.total_points_needed_before_selection
        )
        processor_output["training"] = [False] * len(videos)
        processor_output["original_num_query_points"] = original_num_query_points
        
        if return_visualization_info:
            processor_output["visualization_frames"] = [videos[idx]["frames"][start_indices[idx]:start_indices[idx] + processor_output["labels"][idx].shape[1]] for idx in range(len(videos))]
            processor_output["video_fps"] = [videos[idx]["fps"] for idx in range(len(videos))]
        
        return processor_output

class QueryPredictorEvaluationProcessorTransformWrapper:
    def __init__(
        self,
        processor,
    ):
        self.processor = processor
        
    def __call__(self, examples, return_visualization_info=False, start_indices=None):
        if isinstance(examples, dict):
            videos = examples["video"]
            tracks = examples["tracks"]
            query_points = examples["query_points"]
            visibles = examples["visibles"]
            camera_motion = examples.get("camera_motion")
            text_inputs = examples.get("text")
        elif isinstance(examples, list):
            videos = [i["video"] for i in examples]
            tracks = [i["tracks"] for i in examples]
            query_points = [i["query_points"] for i in examples]
            visibles = [i["visibles"] for i in examples]
            camera_motion = [i.get("camera_motion") for i in examples]
            text_inputs = [i.get("text") for i in examples]
        
        if start_indices is None:
            start_indices = [0] * len(videos)
        images = [v["frames"][start_indices[i]] for i, v in enumerate(videos)]

        processor_output = self.processor(
            images=images,
            tracks=tracks,
            query_points=query_points,
            visibles=visibles,
            camera_motion=camera_motion,
            text_inputs=text_inputs,
        )
        processor_output["training"] = [False] * len(videos)
        if return_visualization_info:
            processor_output["visualization_images"] = [v["frames"][start_indices[i]] for i, v in enumerate(videos)]
        return processor_output
