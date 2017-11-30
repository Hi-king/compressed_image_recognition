import argparse
import random

import chainer
import numpy
import pipe
import os

from PIL import Image
import compressed_image_recoginition
import scipy.misc
import tqdm


def main(args: argparse.Namespace):
    train_mnist_dataset, test_mnist_dataset = chainer.datasets.get_mnist(withlabel=True, ndim=2)
    unpadded_mnist_dataset = compressed_image_recoginition.datasets.MnistCompressedBinaryDataset(base_dataset=test_mnist_dataset,
                                                                                  image_format=args.format)
    test_mnist_dataset = compressed_image_recoginition.datasets.PaddedDataset(unpadded_mnist_dataset)

    model = compressed_image_recoginition.models.load_model(args)

    chainer.serializers.load_npz(args.model_weight, model)

    data, label = test_mnist_dataset[0]
    print(data.shape)

    predicted = model(numpy.array([data]))
    print("true label: {}".format(label))
    print(predicted.data)
    compressed_image_recoginition.datasets.MnistCompressedBinaryDataset.save_formatted(data, "test.jpg")

    # Visualize
    predicteds = model.predict_all_steps(numpy.array([data]))
    from matplotlib import pyplot
    print(predicteds.shape)
    pyplot.imshow(
        scipy.misc.imresize(predicteds.data, (predicteds.data.shape[0], predicteds.data.shape[1] * 10))
    )
    # pyplot.show()
    pyplot.savefig("visualization.png")

    # Glitchey
    for target in tqdm.tqdm(list(range(10))):
        indices = enumerate(test_mnist_dataset) \
                  | pipe.where(lambda i_img_label: i_img_label[1][1] == target) \
                  | pipe.select(lambda i_img_label: i_img_label[0]) \
                  | pipe.take(10) \
                  | pipe.as_list()
        dataset = [test_mnist_dataset[i][0] for i in indices]
        raw_dataset = [unpadded_mnist_dataset[i][0] for i in indices]
        #
        # | pipe.select(
        #     lambda img_label: compressed_image_recoginition.datasets.MnistCompressedBinaryDataset.convert(img_label[0],
        #                                                                                                   args.format)) \
        #           | pipe.take(10)
        for n_glitchy in [0, 1, 2, 5, 10, 20, 50]:
            directory = os.path.join('visualization', 'glitchy{}'.format(n_glitchy), str(target))
            os.makedirs(directory, exist_ok=False)
            for num, (data, raw_data) in enumerate(zip(dataset, raw_dataset)):
                for _ in range(n_glitchy):
                    i = random.randint(0, raw_data.shape[0]-1)
                    data[i] = 0
                    raw_data[i] = 0
                predicted = model(numpy.array([data]))
                result = (numpy.argmax(predicted.data) == target)
                compressed_image_recoginition.datasets.MnistCompressedBinaryDataset.save_formatted(
                    raw_data,
                    os.path.join(directory, '{}_{}.jpg'.format(num, result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_weight")
    parser.add_argument("--format", type=str, default='JPEG')
    parser.add_argument("--gpu", type=int, default=-1)
    compressed_image_recoginition.models.model_args_parser(parser)
    args = parser.parse_args()
    main(args)
