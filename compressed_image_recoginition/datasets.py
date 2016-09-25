# -*- coding: utf-8 -*-
from chainer.dataset import dataset_mixin
import io
import numpy
from PIL import Image


class MnistCompressedBinaryDataset(dataset_mixin.DatasetMixin):
    def __init__(self, base_dataset: dataset_mixin.DatasetMixin, image_format: str = 'JPEG'):
        self._base = base_dataset
        self._format = image_format

    def __len__(self):
        return len(self._base)

    def get_example(self, i) -> (numpy.ndarray, int):
        data, label = self._base[i]

        with io.BytesIO() as f:
            image = Image.fromarray((data * 256).astype(numpy.uint8))
            image.save(fp=f, format=self._format)
            d = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8).astype(numpy.int32)
            return d, label


class HeadDataset(dataset_mixin.DatasetMixin):
    def __init__(self, head: int, base_dataset: dataset_mixin.DatasetMixin):
        self._base = base_dataset
        self._head = head

    def __len__(self):
        return min(len(self._base), self._head)

    def get_example(self, i):
        return self._base[i]
