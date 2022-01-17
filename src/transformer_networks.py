import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable

from transformer_performer_layers import (TransformerPerformerDecoderLayer,
                                          TransformerPerformerEncoderLayer)


class FixedPositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len):
        super().__init__()

        inv_freq = 1. / (10000**(torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x.device)


class Performer(nn.Module):

    def __init__(
        self,
        XE,
        dropout,
        input_dropout,
        input_dropout_rate,
        frozen_trial,
        batch_size,
        n_linear_layers,
        nystrom_attention,
        n_performer_layers,
        gene_dimension,
        gene_length,
        weather_length,
    ):
        self.XE = XE
        if self.XE:
            out = 7
        else:
            out = 1
        self.DIMENSION = gene_dimension
        super(Performer, self).__init__()
        self.n_linear_layers = n_linear_layers
        self.n_performer_layers = n_performer_layers
        self.nystrom_attention = nystrom_attention
        self.normalization = 1
        self.frozen_trial = frozen_trial

        self.input_dropout = input_dropout
        self.input_dropout_rate = input_dropout_rate

        # Identity function
        self.to_cls_token = nn.Identity()

        # Performer specific layers

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, gene_length + 1, self.DIMENSION))

        # Create cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.DIMENSION))

        # Linear specific layers

        self.linear_layers = []
        self.linear_dropouts = []
        self.linear_norm_layers = []

        # Set dropout rate
        self.dropout_rate = dropout

        self.performer_head_dims = int(
            frozen_trial.suggest_categorical("gene_performer_head_dims",
                                             [2, 4, 8]))

        self.n_performer_heads = int(
            frozen_trial.suggest_categorical("n_gene_performer_heads",
                                             [4, 8, 16]))

        self.gene_performer_ff_size = int(
            frozen_trial.suggest_categorical("n_gene_performer_ff_size",
                                             [32, 64, 128]))

        # Nystrom attention does not use kernels to approximate functions. No need to search through them.
        if self.nystrom_attention:
            self.generalized_attention = 0
            self.n_performer_landmarks = int(
                frozen_trial.suggest_categorical("n_gene_performer_landmarks",
                                                 [32, 64, 128, 256, 512]))
        else:
            self.generalized_attention = frozen_trial.suggest_int(
                "generalized_attention", 0, 1)
            self.n_performer_landmarks = 0

        self.performer = torch.nn.TransformerEncoder(
            TransformerPerformerEncoderLayer(
                d_model=self.DIMENSION,
                nystrom_attention=self.nystrom_attention,
                nhead=self.n_performer_heads,
                dim_feedforward=self.gene_performer_ff_size,
                dim_head=self.performer_head_dims,
                dropout=self.dropout_rate,
                activation='gelu',
                generalized_attention=self.generalized_attention,
                n_landmarks=self.n_performer_landmarks),
            num_layers=self.n_performer_layers,
            norm=torch.nn.LayerNorm(self.DIMENSION))

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

        self.norm = nn.LayerNorm(output_dim)
        self.cl1 = nn.Linear(self.DIMENSION + self.weather_out_size,
                             output_dim)
        self.classifier = nn.Linear(output_dim, out)

        # Assigning the layers as class variab
        for idx, layer in enumerate(self.linear_layers):
            setattr(self, "fc{}".format(idx), layer)

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

    def get_optuna_params(self):
        return {
            'n_gene_performer_heads': self.n_performer_heads,
            'gene_performer_head_dim': self.performer_head_dims,
            'gene_performer_ff_size': self.gene_performer_ff_size,
            'gene_performer_generalized_attention': self.generalized_attention,
        }

    def linear_forward(self, x):
        x = self.l_linear(x)
        return x

    def performer_forward(self, x):
        # Run through the performer
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = x.permute(1, 0, 2)
        x = self.performer(x).permute(1, 0, 2)

        # Extract cls token
        x = self.to_cls_token(x[:, 0])

        return x

    def weather_forward(self, z):
        for layer, norm, dropout in zip(self.linear_layers[:-1],
                                        self.linear_norm_layers,
                                        self.linear_dropouts[:-1]):
            z = F.relu(layer(z))
            z = norm(z)
            z = dropout(z)

        z = F.relu(self.linear_layers[-1](z))
        z = self.linear_dropouts[-1](z)
        return z

    def forward(self, batch):
        btch = batch[0]
        x = btch["input_genome"]
        z = btch["input_weather"].float()

        # Do dropout at input if we're doing augmentation
        if self.input_dropout:
            x = self.dropout_in(x)

        z = self.weather_forward(z)

        # Run through performer
        x = self.performer_forward(x)

        x = torch.cat((x, z), 1)
        # x = torch.cat((x.view(x.shape[0], -1), z), 1)

        # Interpret concatenated data
        x = F.relu(self.cl1(x))

        # Layer normalize
        x = self.norm(x)

        # Classify
        x = self.classifier(self.d1(x))
        return torch.sigmoid(x)


class HistoricalPerformer(nn.Module):

    def __init__(
        self,
        XE,
        dropout,
        input_dropout,
        input_dropout_rate,
        frozen_trial,
        batch_size,
        nystrom_attention,
        n_gene_performer_layers,
        n_weather_performer_layers,
        gene_dimension,
        gene_length,
        weather_length,
    ):
        self.XE = XE
        if self.XE:
            out = 7
        else:
            out = 1
        self.GENE_DIMENSION = gene_dimension
        self.WEATHER_DIMENSION = 2
        self.nystrom_attention = nystrom_attention
        self.n_gene_layers = n_gene_performer_layers
        self.n_weather_layers = n_weather_performer_layers

        self.input_dropout = input_dropout
        self.input_dropout_rate = input_dropout_rate

        super(HistoricalPerformer, self).__init__()
        self.frozen_trial = frozen_trial

        # Identity function
        self.to_cls_token = nn.Identity()

        # We upsample the weather data
        self.wt_upsample_size = self.GENE_DIMENSION
        self.wt_upsample = nn.Linear(self.WEATHER_DIMENSION,
                                     self.wt_upsample_size,
                                     bias=False)

        # Positional embedding for weather data
        # We use the same embedding as descrbed in Attention is all you need
        self.wt_pos_embedding = nn.Parameter(
            torch.randn(1, gene_length + weather_length + 1,
                        self.GENE_DIMENSION))
        #self.wt_pos_embedding = FixedPositionalEmbedding(dim=2, max_seq_len=1000)

        # Positional embedding for gene
        self.gn_pos_embedding = nn.Parameter(
            torch.randn(1, gene_length + 1, self.GENE_DIMENSION))

        # Create cls_token
        self.gene_cls_token = nn.Parameter(
            torch.randn(1, 1, self.GENE_DIMENSION))
        self.weather_cls_token = nn.Parameter(
            torch.randn(1, 1, self.wt_upsample_size))

        # Set dropout rate
        self.dropout_rate = dropout

        self.n_gene_performer_heads = 16
        self.gene_performer_head_dims = 2
        self.gene_performer_ff_size = 128
        self.gene_performer_generalized_attention = 1
        self.gene_performer_dropout = 0.29573523437437194

        # We build the performer based on the best architecture found in Performer search
        self.gene_performer = torch.nn.TransformerEncoder(
            TransformerPerformerEncoderLayer(
                d_model=self.GENE_DIMENSION,
                nystrom_attention=0,
                nhead=self.n_gene_performer_heads,
                dim_feedforward=self.gene_performer_ff_size,
                dim_head=self.gene_performer_head_dims,
                dropout=self.gene_performer_dropout,
                activation='gelu',
                generalized_attention=self.
                gene_performer_generalized_attention,
                n_landmarks=0),
            num_layers=1,
            norm=torch.nn.LayerNorm(self.GENE_DIMENSION))

        self.weather_performer_head_dims = int(
            frozen_trial.suggest_categorical("weather_performer_head_dims",
                                             [2, 4, 8]))

        self.n_weather_performer_heads = int(
            frozen_trial.suggest_categorical("n_weather_performer_heads",
                                             [2, 4, 8]))

        self.weather_performer_ff_size = int(
            frozen_trial.suggest_categorical("n_weather_performer_ff_size",
                                             [32, 64, 128]))

        # Nystrom attention does not use kernels to approximate functions. No need to search through them.
        if self.nystrom_attention:
            self.weather_performer_generalized_attention = 0
            self.n_weather_performer_landmarks = int(
                frozen_trial.suggest_categorical(
                    "n_weather_performer_landmarks", [32, 64, 128, 256, 512]))
        else:
            self.weather_performer_generalized_attention = frozen_trial.suggest_int(
                "weather_generalized_attention", 0, 1)
            self.n_weather_performer_landmarks = 0

        self.weather_performer = torch.nn.TransformerEncoder(
            TransformerPerformerEncoderLayer(
                d_model=self.wt_upsample_size,
                nystrom_attention=self.nystrom_attention,
                nhead=self.n_weather_performer_heads,
                dim_feedforward=self.weather_performer_ff_size,
                dim_head=self.weather_performer_head_dims,
                dropout=self.dropout_rate,
                activation='gelu',
                generalized_attention=self.
                weather_performer_generalized_attention,
                n_landmarks=self.n_weather_performer_landmarks),
            num_layers=self.n_weather_layers,
            norm=torch.nn.LayerNorm(self.wt_upsample_size))

        # Find output dimension
        output_dim = int(frozen_trial.suggest_int("n_units_output", 1, 128))

        # Create layer norm
        self.norm = nn.LayerNorm(output_dim)

        # Create layer to interpret the concatenated model outputs
        self.cl1 = nn.Linear(self.GENE_DIMENSION + self.wt_upsample_size,
                             output_dim)

        # Create final classification layer
        self.classifier = nn.Linear(output_dim, out)

        # Dropouts
        self.d1 = nn.Dropout(self.dropout_rate)

        # If we are doing data augmentation through dropout at input
        if self.input_dropout:
            self.dropout_in_gene = nn.Dropout(self.input_dropout_rate)
            self.dropout_in_wt = nn.Dropout(self.input_dropout_rate)

    def get_optuna_params(self):
        return {
            'n_gene_performer_heads':
            self.n_gene_performer_heads,
            'gene_performer_head_dim':
            self.gene_performer_head_dims,
            'gene_performer_ff_size':
            self.gene_performer_ff_size,
            'gene_performer_generalized_attention':
            self.gene_performer_generalized_attention,
            'n_wt_performer_heads':
            self.n_weather_performer_heads,
            'wt_performer_head_dim':
            self.weather_performer_head_dims,
            'wt_performer_ff_size':
            self.weather_performer_ff_size,
            'wt_performer_generalized_attention':
            self.weather_performer_generalized_attention,
        }

    def gene_forward(self, x):

        b, n, _ = x.shape

        # Add cls token
        cls_tokens = repeat(self.gene_cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x += self.gn_pos_embedding[:, :(n + 1)]

        # Run model
        x = x.permute(1, 0, 2)
        x = self.gene_performer(x).permute(1, 0, 2)

        # Extract cls token
        x = self.to_cls_token(x[:, 0])

        return x

    def weather_forward(self, z):
        # Run through the performer
        b, n, _ = z.shape

        # Add cls token
        cls_tokens = repeat(self.weather_cls_token, "() n d -> b n d", b=b)
        z = torch.cat((cls_tokens, z), dim=1)

        # Add positional encoding
        # z += self.wt_pos_embedding(z)
        z += self.wt_pos_embedding[:, :(n + 1)]

        # Run model
        z = z.permute(1, 0, 2)
        z = self.weather_performer(z).permute(1, 0, 2)

        # Extract cls token
        z = self.to_cls_token(z[:, 0])

        return z

    def forward(self, batch):
        # Extract batch
        btch = batch[0]

        # Find the genome
        x = btch["input_genome"]

        # Do augmentation
        if self.input_dropout:
            x = self.dropout_in_gene(x)

        # Run through gene performer
        x = self.gene_forward(x)

        # Extract weather data
        z1 = btch["air_temp"].unsqueeze(dim=1).float()
        z2 = btch["precip"].unsqueeze(dim=1).float()

        # Concat to crate a single tensor
        z = torch.cat((z1, z2), axis=1)

        # Permute tensor to fit our model
        z = z.permute(0, 2, 1)

        # Upsample weather
        z = self.wt_upsample(z)

        if self.input_dropout:
            z = self.dropout_in_wt(z)

        # Run through weather performer
        z = self.weather_forward(z)

        # Concat x and z
        x = torch.cat((x, z), 1)

        # Interpret concatenated data
        x = F.relu(self.cl1(x))

        # Layer normalize
        x = self.norm(x)

        # Classify
        x = self.classifier(self.d1(x))

        return torch.sigmoid(x)


class MultimodalPerformer(nn.Module):

    def __init__(
        self,
        XE,
        dropout,
        input_dropout,
        input_dropout_rate,
        frozen_trial,
        batch_size,
        nystrom_attention,
        n_performer_layers,
        separate_embedding,
        gene_dimension,
        gene_length,
        weather_length,
    ):
        self.XE = XE
        if self.XE:
            out = 7
        else:
            out = 1
        self.DIMENSION = gene_dimension
        super(MultimodalPerformer, self).__init__()
        self.n_performer_layers = n_performer_layers
        self.nystrom_attention = nystrom_attention
        self.normalization = 1
        self.frozen_trial = frozen_trial
        self.separate_embedding = separate_embedding

        self.input_dropout = input_dropout
        self.input_dropout_rate = input_dropout_rate

        # Identity function
        self.to_cls_token = nn.Identity()

        self.wt_upsample_size = self.DIMENSION

        if self.separate_embedding:
            print("Using separate embeddings")
            self.g_pos_embedding = nn.Parameter(
                torch.randn(1, gene_length + 1, self.DIMENSION))
            self.w_pos_embedding = FixedPositionalEmbedding(
                self.DIMENSION, 200)
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, gene_length + weather_length + 1,
                            self.DIMENSION))

        # Linear upsampling layers
        self.upsample_z = nn.Linear(2, self.DIMENSION, bias=False)

        # Create cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.DIMENSION))

        # Set dropout rate
        self.dropout_rate = dropout

        self.performer_head_dims = int(
            frozen_trial.suggest_categorical("gene_performer_head_dims",
                                             [2, 4, 8]))

        self.n_performer_heads = int(
            frozen_trial.suggest_categorical("n_gene_performer_heads",
                                             [4, 8, 16]))

        self.performer_ff_size = int(
            frozen_trial.suggest_categorical("n_gene_performer_ff_size",
                                             [32, 64, 128]))

        # Nystrom attention does not use kernels to approximate functions. No need to search through them.
        if self.nystrom_attention:
            self.generalized_attention = 0
            self.n_performer_landmarks = int(
                frozen_trial.suggest_categorical("n_gene_performer_landmarks",
                                                 [32, 64, 128, 256, 512]))
        else:
            self.generalized_attention = frozen_trial.suggest_int(
                "generalized_attention", 0, 1)
            self.n_performer_landmarks = 0

        self.performer = torch.nn.TransformerEncoder(
            TransformerPerformerEncoderLayer(
                d_model=self.DIMENSION,
                nystrom_attention=self.nystrom_attention,
                nhead=self.n_performer_heads,
                dim_feedforward=self.performer_ff_size,
                dim_head=self.performer_head_dims,
                dropout=self.dropout_rate,
                activation='gelu',
                nb_features=
                300,  #Since we are combining both weather and gene data, we should strive for perfect attention
                generalized_attention=self.generalized_attention,
                n_landmarks=self.n_performer_landmarks),
            num_layers=self.n_performer_layers,
            norm=torch.nn.LayerNorm(self.DIMENSION))

        # Dropouts
        self.d1 = nn.Dropout(self.dropout_rate)

        # If we are doing data augmentation through dropout at input
        if self.input_dropout:
            self.dropout_in_gene = nn.Dropout(self.input_dropout_rate)
            self.dropout_in_wt = nn.Dropout(self.input_dropout_rate)

        # Layer Norm
        self.norm = nn.LayerNorm(self.DIMENSION)

        # Prediction Layer
        self.predictor = nn.Linear(self.DIMENSION, out)

    def get_optuna_params(self):
        return {
            'n_performer_heads': self.n_performer_heads,
            'performer_head_dim': self.performer_head_dims,
            'performer_ff_size': self.performer_ff_size,
            'performer_generalized_attention': self.generalized_attention,
        }

    def performer_forward(self, x, z):
        # Run through the performer

        if self.separate_embedding:
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.g_pos_embedding[:, :(n + 1)]
            z += self.w_pos_embedding(z)
            # Add cls token

        # Concatenate weather to gene
        x = torch.cat((x, z), 1)

        if not self.separate_embedding:
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            # Add positional embedding
            x += self.pos_embedding[:, :(n + 1)]

        # Make input fit with pytorch performers
        x = x.permute(1, 0, 2)

        # Make output of performer fit with traditional expectations
        x = self.performer(x).permute(1, 0, 2)

        # Extract cls token
        x = self.to_cls_token(x[:, 0])

        return x

    def forward(self, batch):
        btch = batch[0]

        # Fetch gene
        x = btch["input_genome"]

        # Do dropout at input if we're doing augmentation
        if self.input_dropout:
            x = self.dropout_in_gene(x)

        # Extract weather data
        z1 = btch["air_temp"].unsqueeze(dim=1).float()
        z2 = btch["precip"].unsqueeze(dim=1).float()

        # Concat to crate a single tensor
        z = torch.cat((z1, z2), axis=1)

        # Do dropout at input if we're doing augmentation
        if self.input_dropout:
            z = self.dropout_in_wt(z)

        # Permute tensor to fit our model
        z = z.permute(0, 2, 1)

        # Upsample to correct size
        z = self.upsample_z(z)

        # Encode that the two weather tensors are a different type of data
        z = z + 1

        # Run through performer
        x = self.performer_forward(x, z)

        # Predict
        x = self.predictor(self.d1(self.norm(x)))
        return torch.sigmoid(x)
