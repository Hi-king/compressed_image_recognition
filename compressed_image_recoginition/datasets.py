# -*- coding: utf-8 -*-
from chainer.dataset import dataset_mixin
import io
import numpy
import tqdm
from PIL import Image


class PaddedDataset(dataset_mixin.DatasetMixin):
    def __init__(self, base_dataset: dataset_mixin.DatasetMixin, max_length=None):
        self.base_dataet = base_dataset
        self.max_length = self.find_max(base_dataset) if max_length is None else max_length
        self.padvalue = -1

    def find_max(self, base_dataset: dataset_mixin.DatasetMixin):
        return max(data.shape[0] for data, _label in tqdm.tqdm(base_dataset, desc="calc maxlen"))

    def __len__(self):
        return len(self.base_dataet)

    def get_example(self, i) -> (numpy.ndarray, int):
        base, label = self.base_dataet[i]
        data = numpy.ones((self.max_length,), dtype=base.dtype)
        data[:min(base.shape[0], data.shape[0])] = base[:min(base.shape[0], data.shape[0])]
        return data, label


class MnistCompressedBinaryDataset(dataset_mixin.DatasetMixin):
    @classmethod
    def convert(cls, base_image, format):
        with io.BytesIO() as f:
            image_array = (base_image * 256).astype(numpy.uint8)
            if format == "npy":
                f.write(image_array.binary_repr())
                image_array.tofile(f)
            else:
                image = Image.fromarray(image_array)
                image.save(fp=f, format=format)
            d = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8).astype(numpy.int32)
            return d

    @classmethod
    def save_formatted(cls, datum, path):
        binary_data = datum.astype(numpy.uint8).tobytes()
        with open(path, "wb+") as f:
            f.write(binary_data)

    def __init__(self, base_dataset: dataset_mixin.DatasetMixin, image_format: str = 'JPEG'):
        self._base = base_dataset
        self._format = image_format

    def __len__(self):
        return len(self._base)

    def get_example(self, i) -> (numpy.ndarray, int):
        data, label = self._base[i]
        return self.convert(data, self._format), label


class HeadDataset(dataset_mixin.DatasetMixin):
    def __init__(self, head: int, base_dataset: dataset_mixin.DatasetMixin):
        self._base = base_dataset
        self._head = head

    def __len__(self):
        return min(len(self._base), self._head)

    def get_example(self, i):
        return self._base[i]
