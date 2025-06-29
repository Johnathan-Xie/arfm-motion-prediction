from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk, Video, Array, Value

from filtering_utils import topk_track_movement
from video_utils import camera_motion_detection
from fractions import Fraction

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--input_dataset", type=str, default=None)
    parser.add_argument("--storage_location", type=str, default="remote")
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--top_k", type=str, default=100)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--writer_batch_size", type=int, default=32)
    parser.add_argument("--dataset_save_name", type=str, default=None)
    parser.add_argument("--add_camera_motion_indicator", default=False, action="store_true")
    parser.add_argument("--add_movement_indicator", default=False, action="store_true")
    parser.add_argument("--frame_rate_target", type=int, default=None)
    args = parser.parse_args()
    return args

def get_dataset(dataset_name, storage_location, decode_video=False):
    if storage_location == "remote":
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_from_disk(dataset_name)
    if decode_video:
        dataset = dataset.cast_column("video", Video())
    
    feature_indicator_split = list(dataset.keys())[0]
    track_array_columns = ["tracks", "visibles"]
    for c in track_array_columns:
        if dataset[feature_indicator_split].features[c] == Value(dtype="binary", id=None):
            dataset = dataset.cast_column(c, Array())
    return dataset

def process_fn(
    example,
    add_camera_motion_indicator=False,
    add_movement_indicator=False,
    frame_rate_mapping_fn=None,
    movement_indicator_skip=5,
    movement_indicator_top_k=100,
):
    output = {}
    if add_movement_indicator:
        output["movement"] = topk_track_movement(
            tracks=example["tracks"],
            visibles=example["visibles"],
            k=movement_indicator_top_k,
            skip=movement_indicator_skip,
        )
    video_feature = Video()
    if add_camera_motion_indicator:
        output["camera_motion"] = camera_motion_detection(example["video"]["frames"])
        example["video"] = video_feature.encode_example(example["video"])
    if frame_rate_mapping_fn is not None:
        example["video"]["fps"] = frame_rate_mapping_fn(example["video"]["fps"])
        output["video"] = video_feature.encode_example(example["video"])
    return output

def main():
    args = parse_args()
    dataset = get_dataset(args.input_dataset, args.storage_location, decode_video=args.add_camera_motion_indicator or args.frame_rate_target)
    
    if args.frame_rate_target is not None:
        frame_rate_mapping_fn = lambda x: Fraction(args.frame_rate_target, 1)
    else:
        frame_rate_mapping_fn = None
    dataset = dataset.map(
        lambda example: process_fn(
            example,
            add_camera_motion_indicator=args.add_camera_motion_indicator,
            add_movement_indicator=args.add_movement_indicator,
            frame_rate_mapping_fn=frame_rate_mapping_fn
        ), num_proc=args.num_proc, writer_batch_size=args.writer_batch_size)
    save_name = args.dataset_save_name or args.input_dataset
    if args.storage_location == "remote":
        dataset.push_to_hub(save_name)
    else:
        dataset.save_to_disk(save_name)

if __name__ == '__main__':
	main()
