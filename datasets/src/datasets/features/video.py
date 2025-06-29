import os
import av
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import warnings
import gc

from ..table import array_cast
from ..utils.file_utils import xopen
from ..utils.py_utils import no_op_if_value_is_null

def frames_to_bytes(frames, codec_name, fps, pix_fmt, video_format='mp4') -> bytes:
    if codec_name == "hevc":
        codec_name = "h264"
    output_bytes = BytesIO()
    output_video = av.open(output_bytes, 'w', format=video_format)
    out_stream = output_video.add_stream(codec_name, fps, options={"x265-params": "log-level=0", "loglevel": "quiet"})
    
    out_stream.height = frames.shape[1]
    out_stream.width = frames.shape[2]
    
    out_stream.pix_fmt = pix_fmt

    for frame in frames:
        out_frame = av.VideoFrame.from_ndarray(frame)  # Note: to_image and from_image is not required in this specific example.
        out_packet = out_stream.encode(out_frame)  # Encode video frame
        output_video.mux(out_packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).
    # Flush the encoder
    out_packet = out_stream.encode(None)
#    out_stream.close()
    output_video.mux(out_packet)

    output_video.close()
    output = output_bytes.getvalue()
    output_bytes.close()
    return output

def bytes_to_frames(input_video, input_stream, start_frame=None, frames_to_read=None):
    frames = []
    for frame in input_video.decode(input_stream):
        frames.append(frame)
    return np.stack([np.array(frame.to_image()) for frame in input_video.decode(input_stream)])

TIME_DEBUG = True

@dataclass
class Video:
    """Image [`Feature`] to read image data from an image file.

    Input: The Image feature accepts as input:
    - 
    - A `str`: Absolute path to the image file (i.e. random access is allowed).
    - An `np.ndarray`: NumPy array representing a video frames x height x width x 3.
    
    Args:
        mode (`str`, *optional*):
            The mode to convert the image to. If `None`, the native mode of the image is used.
        decode (`bool`, defaults to `True`):
            Whether to decode the image data. If `False`,
            returns the underlying dictionary in the format `{"path": image_path, "bytes": image_bytes}`.
    """

    mode: Optional[str] = None
    #codec: Optional[str] = "h264"
    #fps: Optional[int] = 30
    #pix_fmt: Optional[str] = "yuv420p"
    decode: bool = True
    max_frames: int = None
    # Automatically constructed
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Video", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, dict]) -> bytes:
        """Encode example into a format for Arrow.

        Args:
            value (`dict`):
                Data passed as input to Video feature. Should have the following keys
                    - path: path to the video file
                    
        Returns:
            `dict` with "path", "bytes" fields
        """
        if isinstance(value, str):
            with open(value, "rb") as f:
                video_bytes = BytesIO(f.read())
            output = video_bytes.getvalue()
            video_bytes.close()
            return output
        elif isinstance(value, dict):
            return frames_to_bytes(**value)
        else:
            raise ValueError(f"value must be of type str or dict, but is of type {type(value)}")
        
    def decode_example(self, value, token_per_repo_id=None) -> np.ndarray:
        """Decode example image file into image data.

        Args:
            value (`str` or `dict`):
                A string with the absolute image file path, a dictionary with
                keys:

                - `path`: String with absolute or relative image file path.
                - `bytes`: The bytes of the image file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                image files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`).

        Returns:
            `np.ndarray`
        """
        try:
            if isinstance(value, str):
                file = value
            elif isinstance(value, bytes):
                file = BytesIO(value)
            else:
                if value.get("bytes") is not None:
                    file = BytesIO(value["bytes"])
                else:
                    file = value["path"]
            
            input_video = av.open(file)
            input_stream = input_video.streams.video[0]

            frames = np.stack([np.array(frame.to_image()) for frame in input_video.decode(input_stream)])
                
            return {
                "frames": frames,
                "codec_name": input_stream.codec_context.name,
                "fps": input_stream.codec_context.rate,
                "pix_fmt": input_stream.codec_context.pix_fmt,
            }
        except Exception as e:
            print(e)
            return {
                "frames": None
            }

    def flatten(self) -> Union["FeatureType", Dict[str, "FeatureType"]]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from datasets.features.features import Value

        return (
            self
            if self.decode
            else {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray, pa.ListArray]) -> pa.StructArray:
        """Cast an Arrow array to the Video arrow storage type.
        The Arrow types that can be converted to the Video pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the image bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter
        - `pa.list(*)` - it must contain the image array data

        Args:
            storage (`Union[pa.StringArray, pa.StructArray, pa.ListArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Embed image files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.

        Returns:
            `pa.StructArray`: Array in the Image arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """

        @no_op_if_value_is_null
        def path_to_bytes(path):
            with xopen(path, "rb") as f:
                bytes_ = f.read()
            return bytes_

        paths_list = [os.path.basename(path) if path is not None else None for path in storage.field("path").to_pylist()]
        bytes_array = pa.array(
            [
                (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"]) if x is not None else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        if isinstance(bytes_array, pa.ChunkedArray):
            # This is a rare case where the image files overflow the binary container. We only take the first chunk for now.
            bytes_array = bytes_array.chunks[0]
            num_included = len(bytes_array)
            num_excluded = len(paths_list) - num_included
            warnings.warn(
                f"{num_excluded} video files are not included in the Arrow array because they are larger than expected. " \
                f"You can try reducing the number of video files per shard to fix this " \
                f"The excluded files are: {paths_list[-num_excluded:]}", UserWarning
            )
            paths_list = paths_list[:num_included]
        path_array = pa.array(paths_list, type=pa.string())
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null())
        return array_cast(storage, self.pa_type)