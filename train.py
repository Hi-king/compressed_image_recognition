# -*- coding: utf-8 -*-
import chainer

import compressed_image_recoginition

train_dataset, test_dataset = chainer.datasets.get_mnist(withlabel=True, ndim=2)
dataset = compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=train_dataset)

for i in range(10):
    data, label = dataset[i]
    print(data.shape)
