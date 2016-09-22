# -*- coding: utf-8 -*-
import chainer
import numpy
from chainer.training import extensions

import compressed_image_recoginition

train_dataset, test_dataset = chainer.datasets.get_mnist(withlabel=True, ndim=2)
dataset = compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=train_dataset)
iterator = chainer.iterators.SerialIterator(dataset=dataset, batch_size=1)


def loss(binary, label):
    return chainer.functions.softmax_cross_entropy(model(binary), label)

optimizer = chainer.optimizers.Adam()
model = compressed_image_recoginition.models.Model(vocab_size=256, midsize=10, output_dimention=10)
optimizer.setup(model)

updater = chainer.training.StandardUpdater(iterator, optimizer, device=-1,
                                           loss_func=loss)
trainer = chainer.training.Trainer(updater, (10, 'epoch'), out=".")
trainer.extend(extensions.ProgressBar(update_interval=1))
trainer.run()
