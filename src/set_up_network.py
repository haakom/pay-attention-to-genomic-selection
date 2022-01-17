import torch.nn as nn
from optuna.importance import FanovaImportanceEvaluator, get_param_importances

from convolutional_networks import ConvNet, ResNet
from transformer_networks import (HistoricalPerformer, MultimodalPerformer,
                                  Performer)


def suggest_network(suggester_trial, hparams, network_type: int,
                    batch_size: int, optimizer: str, learning_rate: float,
                    momentum: float, weight_decay: float):
    """
    Creates and returns a network to be optimized
    """
    optuna_dropout_rate = suggester_trial.suggest_uniform(
        "dropout_rate", 0.1, 0.9)

    if network_type == 1:
        network_name = 'Vanilla_CNN'
        optuna_n_conv_layers = suggester_trial.suggest_int(
            "n_conv_layers", 1, 10)

        optuna_n_linear_layers = suggester_trial.suggest_int(
            "n_linear_layers", 1, 6)

        model = ConvNet(
            hparams["cross_entropy"],
            optuna_dropout_rate,
            hparams["dropout_input"],
            hparams["dropout_input_rate"],
            suggester_trial,
            batch_size,
            optuna_n_conv_layers,
            optuna_n_linear_layers,
            hparams["gene_size"],
            hparams["gene_length"],
            hparams["weather_length"],
        )
        hyperparams = {
            'lr': learning_rate,
            'Samplery': hparams["sampler"],
            'batch_size': batch_size,
            'optimizer': optimizer,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'n_linear_layers': optuna_n_linear_layers,
            'n_conv_layers': optuna_n_conv_layers,
            'n_gene_performer_layers': 0.0,
            'n_weather_performer_layers': 0.0,
            'weather_data': 0
        }

    elif network_type == 2:
        network_name = 'ResNet'
        optuna_n_conv_layers = suggester_trial.suggest_int(
            "n_conv_layers", 1, 20)

        optuna_n_linear_layers = suggester_trial.suggest_int(
            "n_linear_layers", 1, 6)

        model = ResNet(
            hparams["cross_entropy"],
            optuna_dropout_rate,
            hparams["dropout_input"],
            hparams["dropout_input_rate"],
            suggester_trial,
            batch_size,
            optuna_n_conv_layers,
            optuna_n_linear_layers,
            hparams["gene_size"],
            hparams["gene_length"],
            hparams["weather_length"],
        )
        hyperparams = {
            'lr': learning_rate,
            'Sampler': hparams["sampler"],
            'batch_size': batch_size,
            'optimizer': optimizer,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'n_linear_layers': optuna_n_linear_layers,
            'n_conv_layers': optuna_n_conv_layers,
            'n_gene_performer_layers': 0.0,
            'n_weather_performer_layers': 0.0,
            'weather_data': 0
        }

    elif network_type == 3:
        network_name = 'Performer'
        optuna_n_linear_layers = suggester_trial.suggest_int(
            "n_linear_layers", 1, 6)
        optuna_n_performer_layers = suggester_trial.suggest_int(
            "n_linear_layers", 1, 4)

        model = Performer(
            hparams["cross_entropy"],
            optuna_dropout_rate,
            hparams["dropout_input"],
            hparams["dropout_input_rate"],
            suggester_trial,
            batch_size,
            optuna_n_linear_layers,
            hparams["nystrom_attention"],
            optuna_n_performer_layers,
            hparams["gene_size"],
            hparams["gene_length"],
            hparams["weather_length"],
        )
        model_hparams = model.get_optuna_params()
        hyperparams = {
            'lr': learning_rate,
            'Sampler': hparams["sampler"],
            'batch_size': batch_size,
            'optimizer': optimizer,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'n_linear_layers': optuna_n_linear_layers,
            'n_conv_layers': 0.0,
            'n_gene_layers': optuna_n_performer_layers,
            'n_gene_heads': model_hparams["n_gene_performer_heads"],
            'gene_head_dim': model_hparams["gene_performer_head_dim"],
            'gene_ff_size': model_hparams["gene_performer_ff_size"],
            'gene_gen_attn':
            model_hparams["gene_performer_generalized_attention"],
            'nystrom_attention': hparams["nystrom_attention"],
            'n_weather_performer_layers': 0.0,
            'weather_data': 0
        }

    elif network_type == 4:
        network_name = 'Historical_Performer'

        optuna_n_weather_performer_layers = suggester_trial.suggest_int(
            'n_weather_performer_layers', 1, 4)

        model = HistoricalPerformer(
            hparams["cross_entropy"],
            optuna_dropout_rate,
            hparams["dropout_input"],
            hparams["dropout_input_rate"],
            suggester_trial,
            batch_size,
            hparams["nystrom_attention"],
            1,
            optuna_n_weather_performer_layers,
            hparams["gene_size"],
            hparams["gene_length"],
            hparams["weather_length"],
        )

        model_hparams = model.get_optuna_params()

        hyperparams = {
            'lr': learning_rate,
            'Sampler': hparams["sampler"],
            'batch_size': batch_size,
            'optimizer': optimizer,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'n_linear_layers': 0.0,
            'n_conv_layers': 0.0,
            'n_gene_layers': 1.0,
            'n_gene_heads': model_hparams["n_gene_performer_heads"],
            'gene_head_dim': model_hparams["gene_performer_head_dim"],
            'gene_ff_size': model_hparams["gene_performer_ff_size"],
            'gene_gen_attn':
            model_hparams["gene_performer_generalized_attention"],
            'n_weather_performer_layers': optuna_n_weather_performer_layers,
            'n_wt_heads': model_hparams["n_wt_performer_heads"],
            'wt_head_dim': model_hparams["wt_performer_head_dim"],
            'wt_ff_size': model_hparams["wt_performer_ff_size"],
            'wt_gen_attn': model_hparams["wt_performer_generalized_attention"],
            'nystrom_attention': hparams["nystrom_attention"],
            'weather_data': 1.0
        }

    elif network_type == 5:
        network_name = ''
        if hparams['separate_embedding']:
            network_name += 'separate_embedding_'
        network_name += 'MultimodalPerformer'
        optuna_n_performer_layers = suggester_trial.suggest_int(
            'n_performer_layers', 1, 4)

        model = MultimodalPerformer(
            hparams["cross_entropy"],
            optuna_dropout_rate,
            hparams["dropout_input"],
            hparams["dropout_input_rate"],
            suggester_trial,
            batch_size,
            hparams["nystrom_attention"],
            optuna_n_performer_layers,
            hparams["separate_embedding"],
            hparams["gene_size"],
            hparams["gene_length"],
            hparams["weather_length"],
        )
        model_hparams = model.get_optuna_params()
        hyperparams = {
            'lr': learning_rate,
            'Sampler': hparams["sampler"],
            'batch_size': batch_size,
            'optimizer': optimizer,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'n_linear_layers': 0.0,
            'n_conv_layers': 0.0,
            'n_performer_layers': optuna_n_performer_layers,
            'n_heads': model_hparams["n_performer_heads"],
            'head_dim': model_hparams["performer_head_dim"],
            'ff_size': model_hparams["performer_ff_size"],
            'gen_attn': model_hparams["performer_generalized_attention"],
            'nystrom_attention': hparams["nystrom_attention"],
            'n_weather_performer_layers': 0.0,
            'weather_data': 1.0
        }

    # Initialize model weights using xavier
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model, hyperparams, network_name
