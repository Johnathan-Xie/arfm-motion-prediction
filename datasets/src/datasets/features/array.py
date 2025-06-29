import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa

from ..table import array_cast

TIME_DEBUG = True

@dataclass
class Array:
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

    # Automatically constructed
    decode: bool = True
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary()})
    _type: str = field(default="Array", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: np.ndarray) -> bytes:
        """Encode example into a format for Arrow.

        Args:
            value (`dict`):
                Data passed as input to Video feature. Should have the following keys
                    - path: path to the video file
                    
        Returns:
            `dict` with "path", "bytes" fields
        """
        if isinstance(value, np.ndarray):
            buffer = BytesIO()
            np.save(buffer, value)
            output = buffer.getvalue()
            buffer.close()
            return output
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
        # Can remove squeeze for future datasets, droid processed wrong
        if isinstance(value, bytes):
            buffer = BytesIO(value)
            array = np.load(buffer)
            #if array.shape[0] == 1:
            #    array = array.squeeze(0)
            return array
        elif isinstance(value, dict):
            buffer = BytesIO(value["bytes"])
            array = np.load(buffer)
            #if array.shape[0] == 1:
            #    array = array.squeeze(0)
            return array
        elif isinstance(value, list):
            return np.array(value)
        else:
            return value
    
    def flatten(self) -> Union["FeatureType"]:
        """If in the decodable state, return the feature itself, otherwise flatten the feature into a dictionary."""
        from datasets.features.features import Value

        return (
            self
            if self.decode
            else Value("binary")
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
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            storage = pa.StructArray.from_arrays([storage], ["bytes"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            
            storage = pa.StructArray.from_arrays([bytes_array], ["bytes"], mask=storage.is_null())
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
        bytes_array = pa.array(
            [
                x["bytes"]
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        storage = pa.StructArray.from_arrays([bytes_array], ["bytes"], mask=bytes_array.is_null())
        return array_cast(storage, self.pa_type)