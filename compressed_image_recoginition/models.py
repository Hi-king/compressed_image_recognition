# -*- coding: utf-8 -*-
import chainer


class LSTMModel(chainer.Chain):
    def __init__(self, vocab_size: int, midsize: int, output_dimention: int, num_lstm_layer: int):
        self.lstm_layer_keys = ["rnn{}".format(i) for i in range(num_lstm_layer)]
        self.lstm_layers = {key: chainer.links.LSTM(midsize, midsize) for key in self.lstm_layer_keys}
        super().__init__(
            word_embed=chainer.links.EmbedID(vocab_size, midsize),
            out_layer=chainer.links.Linear(midsize, output_dimention),
            **self.lstm_layers
        )

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """
        :param x: [batch, bytes]
        """
        self.reset_state()
        outputs = []
        x = self.word_embed(x)
        for byte in chainer.functions.separate(x, axis=1):
            lstm_out = self._forward_lstms(byte)
        out = self.out_layer(lstm_out)
        return out

        # for each_data in chainer.functions.separate(x):
        #     for byte in chainer.functions.separate(each_data):
        #         byte = chainer.functions.expand_dims(byte, axis=0)
        #         lstm_out = self._forward_lstms(byte)
        #     last_output = self.out_layer(lstm_out)
        #     outputs.append(last_output)
        # return chainer.functions.concat(outputs)

    def _forward_lstms(self, x):
        # h = self.word_embed(x)
        h = x
        for key in self.lstm_layer_keys:
            h = getattr(self, key)(h)
        return h

    def reset_state(self):
        for key in self.lstm_layer_keys:
            getattr(self, key).reset_state()


class ConvBNBlock(chainer.Chain):
    def __init__(self, channels, bn):
        super().__init__()
        self.enable_bn = bn
        with self.init_scope():
            self.conv = chainer.links.ConvolutionND(ndim=1, in_channels=channels, out_channels=channels,
                                                    ksize=(3,))
            if self.enable_bn:
                self.BN = chainer.links.BatchNormalization(channels)

    def __call__(self, x: chainer.Variable):
        h = self.conv(x)
        if self.enable_bn:
            h = self.BN(h)
        return chainer.functions.relu(h)


class ConvLSTM(LSTMModel):
    def __init__(self, vocab_size, midsize, output_dimention, num_lstm_layer, bn):
        super().__init__(vocab_size, midsize, output_dimention, num_lstm_layer)
        with self.init_scope():
            self.convolutions = chainer.ChainList(
                ConvBNBlock(midsize, bn=bn),
                ConvBNBlock(midsize, bn=bn),
            )

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        self.reset_state()
        outputs = []
        h = self.word_embed(x)
        h = chainer.functions.swapaxes(h, 1, 2)
        for block in self.convolutions:
            h = block(h)
        h = chainer.functions.swapaxes(h, 1, 2)

        for byte in chainer.functions.separate(h, axis=1):
            lstm_out = self._forward_lstms(byte)
        out = self.out_layer(lstm_out)
        return out


        # for each_data in chainer.functions.separate(h):
        #     for byte in chainer.functions.separate(each_data):
        #         byte = chainer.functions.expand_dims(byte, axis=0)
        #         lstm_out = self._forward_lstms(byte)
        #     last_output = self.out_layer(lstm_out)
        #     outputs.append(last_output)
        # return chainer.functions.concat(outputs)
