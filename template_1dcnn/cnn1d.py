import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def convolution_1d(in_channels: int,
                   out_channels: int,
                   batch_normalization: bool,
                   bias: bool,
                   kernel_size: int = 3,
                   stride: int = 1,
                   activation: torch.nn.Module = torch.nn.ReLU(),
                   no_feat=None,
                   **kwargs,) -> torch.nn.Sequential:
    layer = []
    if no_feat == None:
        conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        )
    else:
        conv = torch.nn.Conv1d(
            in_channels=no_feat,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        )

    layer.append(conv)
    if batch_normalization:
        layer.append(torch.nn.BatchNorm1d(num_features=out_channels))
    if activation:
        layer.append(activation)
    return torch.nn.Sequential(*layer)


def fully_connected(in_features: int,
                    out_features: int,
                    bias: bool,
                    layer_normalization: bool,
                    activation: torch.nn.Module = torch.nn.ReLU(),
                    no_classes = None) -> torch.nn.Sequential:
    layer = []
    if no_classes == None:
        fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    else:
        fc = torch.nn.Linear(in_features=no_classes, out_features=out_features, bias=bias)

    fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    layer.append(fc)
    if layer_normalization:
        layer.append(torch.nn.LayerNorm(normalized_shape=out_features))
    if activation:
        layer.append(activation)
    return torch.nn.Sequential(*layer)


# Convolutional NN setup
BATCH_NORM = True
BIAS = False
ACTIVATION = torch.nn.LeakyReLU()
MULT = 8
CONV_NNPool_SETUP = {
    "layer_in": ["conv_pool", None, 2 * MULT, BATCH_NORM, BIAS, 7, 3, ACTIVATION],
    "layer_2": ["conv_pool", 2 * MULT, 4 * MULT, BATCH_NORM, BIAS, 5, 2, ACTIVATION],
    "layer_3": ["conv_pool", 4 * MULT, 8 * MULT, BATCH_NORM, BIAS, 5, 2, ACTIVATION],
    "layer_4": ["conv_pool", 8 * MULT, 16 * MULT, BATCH_NORM, BIAS, 3, 2, ACTIVATION],
    "layer_5": ["conv_pool", 16 * MULT, 8 * MULT, BATCH_NORM, BIAS, 3, 2, ACTIVATION],
    "layer_6": ["conv_pool", 8 * MULT, 4 * MULT, BATCH_NORM, BIAS, 3, 2, ACTIVATION],
    "layer_7": ["conv_pool", 4 * MULT, 16, BATCH_NORM, BIAS, 3, 2, ACTIVATION]
}

FC_NNPool_SETUP = {
    "layer_8": ["fc", 48, 64, BIAS, BATCH_NORM, ACTIVATION],
    "layer_9": ["fc", 64, 32, BIAS, BATCH_NORM, ACTIVATION],
    "layer_out": ["fc", 32, None, False, False, None],
}


class Convolutional1dNN(torch.nn.Module):
    def __init__(self, no_feat: int, no_classes: int, layer_map: dict):
        super().__init__()
        self.conv_network = []
        for layer in CONV_NNPool_SETUP.keys():
            layer_info = CONV_NNPool_SETUP[layer]
            layer_type = layer_info[0]
            if layer == 'layer_in':
                self.conv_network += layer_map[layer_type](*layer_info[1:], no_feat=no_feat)
            else:
                self.conv_network += layer_map[layer_type](*layer_info[1:])
        self.conv_network = torch.nn.Sequential(*self.conv_network)
        self.fc_network = []
        for layer in FC_NNPool_SETUP.keys():
            layer_info = FC_NNPool_SETUP[layer]
            layer_type = layer_info[0]
            if layer == 'layer_out':
                self.fc_network += layer_map[layer_type](*layer_info[1:], no_classes=no_classes)
            else:
                self.fc_network += layer_map[layer_type](*layer_info[1:])
        self.fc_network = torch.nn.Sequential(*self.fc_network)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = self.conv_network(inputs)
        outs = outs.reshape(inputs.shape[0], -1)
        print(outs.size())
        outs = self.fc_network(outs)
        return outs


def main():

    layer_map = {
        "fc": fully_connected,
        "conv_pool": convolution_1d,
    }

    model = Convolutional1dNN(no_feat=19, no_classes=1, layer_map=layer_map).to('cpu')
    print(summary(model, (19, 1028)))


if __name__ == "__main__":
    main()
