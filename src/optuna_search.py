import os
import random
import time

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, loggers, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from optuna_lightning_module import GenomeModule
from optuna_utils import prepare_study


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def run_trials(
    ttargs,
    cluster=None,
):
    # Try to avoid all trials trying to create the same study at once
    if ttargs.slurm == 1:
        time.sleep(random.randint(1, 30))

    study, h_search_frozen_trial = prepare_study(ttargs)

    study.optimize(lambda trial: objective(
        trial,
        hyperparser=ttargs,
        frozen_trial=h_search_frozen_trial,
    ),
                   n_trials=ttargs.total_num_trials)


def objective(
    trial,
    hyperparser,
    frozen_trial,
):

    # Seed the entire experiment with our set seed
    seed_everything(hyperparser.random_seed)

    if hyperparser.network == 1:
        folder = "CNN/"
    elif hyperparser.network == 2:
        folder = "ResNet/"
    elif hyperparser.network == 3:
        folder = "Performer/"
    elif hyperparser.network == 4:
        folder = "Historical_Performer/"
    elif hyperparser.network == 5:
        folder = "Multimodal_Performer/"

    # Filenames for each trial must be made unique in order to access each checkpoint.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(hyperparser.model_dir, folder,
                     "trial_{}".format(trial.number), "{epoch}"),
        monitor="val_loss",
        save_top_k=1,
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()

    tb_logger = loggers.TensorBoardLogger(
        save_dir=os.path.join(hyperparser.logdir, "tensorboard", folder),
        log_graph=True,
        default_hp_metric=False,
    )
    if hyperparser.one_cycle == 1:
        print('one_cycle')
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Looks like we need one trainer for the tuner
        trainer = pl.Trainer(
            deterministic=True,
            limit_val_batches=0.0,
            auto_scale_batch_size="power",
            gpus=1 if torch.cuda.is_available() else None,
        )
        model = GenomeModule(hyperparser, trial, hyperparser.datasets_dir,
                             frozen_trial)
        new_batch_size = trainer.tune(model, trial_num=trial.number)
        model.hparams.batch_size = new_batch_size
        del (trainer)

        # And then initializing a new one for the lr-finder to work
        trainer = pl.Trainer(
            logger=tb_logger,
            deterministic=True,
            accumulate_grad_batches=1,
            checkpoint_callback=checkpoint_callback,
            limit_val_batches=1.0,
            # Allow pytorch-lightning to find the optimal batch size
            auto_scale_batch_size="power",
            max_epochs=hyperparser.num_epochs,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[
                lr_monitor,
                metrics_callback,
            ],
        )

        lr_finder = trainer.tuner.lr_find(model,
                                          trial_num=trial.number,
                                          min_lr=1e-8)
        suggested_lr = lr_finder.suggestion(skip_begin=30)
        hyperparser.lr = suggested_lr
        model.hparams.lr = suggested_lr

    else:
        print('regular')
        trainer = pl.Trainer(
            deterministic=True,
            accumulate_grad_batches=1,
            logger=tb_logger,
            checkpoint_callback=checkpoint_callback,
            limit_val_batches=1.0,
            # checkpoint_callback=checkpoint_callback,
            # Allow pytorch-lightning to find the optimal batch size
            auto_scale_batch_size="power",
            max_epochs=hyperparser.num_epochs,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[
                metrics_callback,
            ],
        )
        model = GenomeModule(hyperparser, trial, hyperparser.datasets_dir,
                             frozen_trial)
        if hyperparser.find_batch_size:
            trainer.tune(model)

    if hyperparser.hyper_search:
        print("Searching hyperparameter space")
    else:
        print(
            "Using best model params from hyperparameter search and crossvalidating"
        )
    trainer.fit(model)

    if not hyperparser.hyper_search:
        # Pytorch lightning loads the best weights for us
        trainer.test(ckpt_path='best')

    return metrics_callback.metrics[-1]["val_loss"].item()
