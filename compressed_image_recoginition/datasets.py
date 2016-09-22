# -*- coding: utf-8 -*-
from chainer.dataset import dataset_mixin
import io
import numpy
from PIL import Image

class MnistCompressedBinaryDataset(dataset_mixin.DatasetMixin):
    def __init__(self, base_dataset: dataset_mixin.DatasetMixin):
        self._base = base_dataset

    def __len__(self):
        return len(self._base)

    def get_example(self, i) -> (numpy.ndarray, int):
        data, label = self._base[i]

        with io.BytesIO() as f:
            image = Image.fromarray((data * 256).astype(numpy.uint8))
            image.save(fp=f, format='JPEG')
            d = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8)
            return d, label
