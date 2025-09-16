# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from io import BytesIO
from typing import Any

from PIL import Image


class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError


class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f).convert(mode="RGB")


class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


class DenseTargetDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        return Image.open(f)
