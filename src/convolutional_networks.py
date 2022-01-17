import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.autograd import Variable


class ResidualBlock(nn.Module):

    def __init__(
        self,
        inchannels: int,
        channels: int,
        stride: int = 2,
        kernel_size=6,
    ) -> None:
        super(ResidualBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inchannels, channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size)
        self.bn2 = nn.BatchNorm1d(channels)
        self.stride = stride

        # Padding for convolutional layers
        self.m0 = nn.ZeroPad2d((8, 8, 0, 0))

        # Padding for identity1
        self.m1 = nn.ZeroPad2d((1, 0, 0, 0))

        # Padding for identity2
        self.m2 = nn.ZeroPad2d((0, 1, 0, 0))

        # Define downsampling layers
        self.downsample1 = nn.Sequential(
            nn.Conv1d(inchannels, channels, 1, 2, bias=False), )

        # Set the weights of downsample layers to 1
        with torch.no_grad():
            self.downsample1[0].weight = nn.Parameter(
                torch.ones_like(self.downsample1[0].weight))

        # Turn of grad for downsample layers
        for param in self.downsample1.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Pad identity1 to the right and identity 2 to the left to avoid loosing any signal during downsampling
        identity1 = self.m1(identity)
        identity2 = self.m2(identity)

        # Downample. The downsample layer has no gradients, so we can reuse it
        identity1 = self.downsample1(identity1)
        identity2 = self.downsample1(identity2)
        identity = identity1 + identity2

        # Pad the input to make it fit with identity downsampling
        out = self.m0(x)

        # Pass through main block
        out = self.conv1(self.m0(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ConvNet(nn.Module):

    def __init__(
        self,
        XE,
        dropout,
        input_dropout,
        input_dropout_rate,
        frozen_trial,
        batch_size,
        n_conv_layers,
        n_linear_layers,
        gene_dimension,
        gene_length,
        weather_length,
    ):
        self.normalization = 1
        self.XE = XE
        if self.XE:
            out = 7
        else:
            out = 1
        super(ConvNet, self).__init__()

        self.conv_layers = []
        self.conv_dropouts = []
        self.conv_norm_layers = []

        self.linear_layers = []
        self.linear_dropouts = []
        self.linear_norm_layers = []

        self.dropout_rate = dropout

        self.DIMENSION = gene_dimension

        conv_input_filters = self.DIMENSION

        self.input_dropout = input_dropout
        self.input_dropout_rate = input_dropout_rate

        # Define conv layers
        for i in range(n_conv_layers):
            conv_output_filters = int(
                frozen_trial.suggest_int("n_conv_filters_l{}".format(i), 16,
                                         256))
            conv_layer = nn.Conv1d(conv_input_filters,
                                   conv_output_filters,
                                   6,
                                   stride=2)
            self.conv_layers.append(conv_layer)
            if i < n_conv_layers - 1:
                if self.normalization == 1:
                    self.conv_norm_layers.append(
                        nn.BatchNorm1d(num_features=conv_output_filters))
            self.conv_dropouts.append(nn.Dropout(self.dropout_rate))
            conv_input_filters = conv_output_filters

        # Find the size of the output of our convolutional layers
        dummy1 = torch.zeros(size=(batch_size, self.DIMENSION, gene_length))
        self.conv_out_size = self.conv_forward(dummy1)
        self.conv_out_size = self.conv_out_size.view(
            self.conv_out_size.shape[0], -1)[1].size()[0]

        input_dim = 2
        # Define linear layers
        for i in range(n_linear_layers):
            output_dim = frozen_trial.suggest_int("n_units_l{}".format(i), 16,
                                                  32)

            self.linear_layers.append(nn.Linear(input_dim, output_dim))
            if i < n_linear_layers - 1:
                if self.normalization == 1:
                    self.linear_norm_layers.append(
                        nn.LayerNorm(normalized_shape=output_dim))
            self.linear_dropouts.append(nn.Dropout(self.dropout_rate))
            input_dim = output_dim
        # self.lin1 = nn.Linear(2, 140)

        # Find the size of the output of the weather network
        dummy2 = torch.zeros(size=(batch_size, 2))
        self.weather_out_size = self.weather_forward(dummy2)[1].size()[0]

        output_dim = int(frozen_trial.suggest_int("n_units_output", 1, 128))

        if self.normalization:
            self.final_norm = nn.LayerNorm(self.conv_out_size +
                                           self.weather_out_size)
        self.cl1 = nn.Linear(self.conv_out_size + self.weather_out_size,
                             output_dim)
        self.classifier = nn.Linear(output_dim, out)

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.conv_layers):
            setattr(self, "conv{}".format(idx), layer)

        if self.normalization == 1:
            # Assigning the normalizations as class variables (PyTorch requirement).
            for idx, norm in enumerate(self.conv_norm_layers):
                setattr(self, "c_norm{}".format(idx), norm)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.conv_dropouts):
            setattr(self, "c_drop{}".format(idx), dropout)

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.linear_layers):
            setattr(self, "fc{}".format(idx), layer)

        if self.normalization == 1:
            # Assigning the normalizations as class variables (PyTorch requirement).
            for idx, norm in enumerate(self.linear_norm_layers):
                setattr(self, "fc_norm{}".format(idx), norm)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.linear_dropouts):
            setattr(self, "f_drop{}".format(idx), dropout)

        # Dropouts
        self.d1 = nn.Dropout(self.dropout_rate)

        # If we are doing data augmentation through dropout at input
        if self.input_dropout:
            self.dropout_in = nn.Dropout(self.input_dropout_rate)

    def conv_forward(self, x):
        if self.normalization == 1:
            for layer, norm, dropout in zip(self.conv_layers[:-1],
                                            self.conv_norm_layers,
                                            self.conv_dropouts[:-1]):
                x = F.relu(layer(x))
                x = norm(x)
                x = dropout(x)
        else:
            for layer, dropout in zip(self.conv_layers[:-1],
                                      self.conv_dropouts[:-1]):
                x = F.relu(layer(x))
                x = dropout(x)
        x = F.relu(self.conv_layers[-1](x))
        x = self.conv_dropouts[-1](x)
        return x

    def weather_forward(self, z):
        if self.normalization == 1:
            for layer, norm, dropout in zip(
                    self.linear_layers[:-1],
                    self.linear_norm_layers,
                    self.linear_dropouts[:-1],
            ):
                z = F.relu(layer(z))
                z = norm(z)
                z = dropout(z)
        else:
            for layer, dropout in zip(self.linear_layers[:-1],
                                      self.linear_dropouts[:-1]):
                z = F.relu(layer(z))
                z = dropout(z)

        z = F.relu(self.linear_layers[-1](z))
        z = self.linear_dropouts[-1](z)
        return z

    def forward(self, batch):
        btch = batch[0]
        x = btch["input_genome"].permute(0, 2, 1)
        z = btch["input_weather"].float()

        # Do dropout at input if we're doing augmentation
        if self.input_dropout:
            x = self.dropout_in(x)

        x = self.conv_forward(x)
        z = self.weather_forward(z)

        # x = torch.reshape(x, (x.shape, -1))

        x = torch.cat((torch.reshape(x, (x.shape[0], -1)), z), 1)
        # x = torch.cat((x.view(x.shape[0], -1), z), 1)
        if self.normalization:
            x = self.final_norm(x)

        # Interpret concatenated data
        x = F.relu(self.cl1(x))

        # Classify
        x = self.classifier(self.d1(x))
        return torch.sigmoid(x)


class ResNet(nn.Module):

    def __init__(
        self,
        XE,
        dropout,
        input_dropout,
        input_dropout_rate,
        frozen_trial,
        batch_size,
        n_conv_layers,
        n_linear_layers,
        gene_dimension,
        gene_length,
        weather_length,
    ):
        self.normalization = 1
        self.XE = XE
        if self.XE:
            out = 7
        else:
            out = 1
        super(ResNet, self).__init__()

        self.conv_layers = []
        self.conv_dropouts = []
        self.conv_norm_layers = []

        self.linear_layers = []
        self.linear_dropouts = []
        self.linear_norm_layers = []

        self.dropout_rate = dropout

        self.DIMENSION = gene_dimension

        conv_input_filters = self.DIMENSION

        self.input_dropout = input_dropout
        self.input_dropout_rate = input_dropout_rate

        # Define conv layers
        for i in range(n_conv_layers):
            conv_output_filters = int(
                frozen_trial.suggest_int("n_conv_filters_l{}".format(i), 16,
                                         256))
            conv_layer = ResidualBlock(conv_input_filters,
                                       conv_output_filters,
                                       kernel_size=6,
                                       stride=2)
            self.conv_layers.append(conv_layer)
            if i < n_conv_layers - 1:
                if self.normalization == 1:
                    self.conv_norm_layers.append(
                        nn.BatchNorm1d(num_features=conv_output_filters))
            self.conv_dropouts.append(nn.Dropout(self.dropout_rate))
            conv_input_filters = conv_output_filters

        # Find the size of the output of our convolutional layers
        dummy1 = torch.ones(size=(batch_size, self.DIMENSION, gene_length))
        self.conv_out_size = self.conv_forward(dummy1)
        self.conv_out_size = self.conv_out_size.view(
            self.conv_out_size.shape[0], -1)[1].size()[0]

        input_dim = 2
        # Define linear layers
        for i in range(n_linear_layers):
            output_dim = frozen_trial.suggest_int("n_units_l{}".format(i), 16,
                                                  32)

            self.linear_layers.append(nn.Linear(input_dim, output_dim))
            if i < n_linear_layers - 1:
                if self.normalization == 1:
                    self.linear_norm_layers.append(
                        nn.LayerNorm(normalized_shape=output_dim))
            self.linear_dropouts.append(nn.Dropout(self.dropout_rate))
            input_dim = output_dim
        # self.lin1 = nn.Linear(2, 140)

        # Find the size of the output of the weather network
        dummy2 = torch.zeros(size=(batch_size, 2))
        self.weather_out_size = self.weather_forward(dummy2)[1].size()[0]

        output_dim = int(frozen_trial.suggest_int("n_units_output", 1, 128))

        if self.normalization:
            self.final_norm = nn.LayerNorm(self.conv_out_size +
                                           self.weather_out_size)
        self.cl1 = nn.Linear(self.conv_out_size + self.weather_out_size,
                             output_dim)
        self.classifier = nn.Linear(output_dim, out)

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.conv_layers):
            setattr(self, "conv{}".format(idx), layer)

        if self.normalization == 1:
            # Assigning the normalizations as class variables (PyTorch requirement).
            for idx, norm in enumerate(self.conv_norm_layers):
                setattr(self, "c_norm{}".format(idx), norm)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.conv_dropouts):
            setattr(self, "c_drop{}".format(idx), dropout)

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.linear_layers):
            setattr(self, "fc{}".format(idx), layer)

        if self.normalization == 1:
            # Assigning the normalizations as class variables (PyTorch requirement).
            for idx, norm in enumerate(self.linear_norm_layers):
                setattr(self, "fc_norm{}".format(idx), norm)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.linear_dropouts):
            setattr(self, "f_drop{}".format(idx), dropout)

        # Dropouts
        self.d1 = nn.Dropout(self.dropout_rate)
        # If we are doing data augmentation through dropout at input
        if self.input_dropout:
            self.dropout_in = nn.Dropout(self.input_dropout_rate)

    def conv_forward(self, x):
        if self.normalization == 1:
            for layer, norm, dropout in zip(self.conv_layers[:-1],
                                            self.conv_norm_layers,
                                            self.conv_dropouts[:-1]):
                x = F.relu(layer(x))
                x = norm(x)
                x = dropout(x)
        else:
            for layer, dropout in zip(self.conv_layers[:-1],
                                      self.conv_dropouts[:-1]):
                x = F.relu(layer(x))
                x = dropout(x)
        x = F.relu(self.conv_layers[-1](x))
        x = self.conv_dropouts[-1](x)
        return x

    def weather_forward(self, z):
        if self.normalization == 1:
            for layer, norm, dropout in zip(
                    self.linear_layers[:-1],
                    self.linear_norm_layers,
                    self.linear_dropouts[:-1],
            ):
                z = F.relu(layer(z))
                z = norm(z)
                z = dropout(z)
        else:
            for layer, dropout in zip(self.linear_layers[:-1],
                                      self.linear_dropouts[:-1]):
                z = F.relu(layer(z))
                z = dropout(z)

        z = F.relu(self.linear_layers[-1](z))
        z = self.linear_dropouts[-1](z)
        return z

    def forward(self, batch):
        btch = batch[0]
        x = btch["input_genome"].permute(0, 2, 1)
        z = btch["input_weather"].float()

        # Do dropout at input if we're doing augmentation
        if self.input_dropout:
            x = self.dropout_in(x)

        x = self.conv_forward(x)
        z = self.weather_forward(z)

        # x = torch.reshape(x, (x.shape, -1))

        x = torch.cat((torch.reshape(x, (x.shape[0], -1)), z), 1)
        # x = torch.cat((x.view(x.shape[0], -1), z), 1)
        if self.normalization:
            x = self.final_norm(x)

        # Interpret concatenated data
        x = F.relu(self.cl1(x))

        # Classify
        x = self.classifier(self.d1(x))
        return torch.sigmoid(x)
