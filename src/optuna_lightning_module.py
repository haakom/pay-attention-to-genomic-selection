import datetime
import os
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pyxis.torch as pxt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error as l1
from torch.optim import SGD, AdamW, Rprop
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.sampler import SubsetRandomSampler

from log_git import get_git_revision_hash
from set_up_network import suggest_network


class GenomeModule(pl.LightningModule):

    def __init__(self,
                 hparams=None,
                 trial=None,
                 data_dir=None,
                 frozen_trial=None):
        super(GenomeModule, self).__init__()
        self.hparams = hparams
        self.trial = trial
        self.frozen_trial = frozen_trial
        self.data_dir = data_dir
        self.batch_size = self.hparams["batch_size"]
        self.best_loss = np.inf
        self.l1 = l1
        self.batch_nb = 0  # This variable determines when the validation logging is run in tensorboard

        # We want to store some test variables
        self.locations = []
        self.xs = []
        self.ys = []
        self.preds = []

        fileDir = os.path.dirname(os.path.abspath(__file__))
        parentDir = os.path.dirname(fileDir)
        parentDir = os.path.dirname(parentDir)

        dataset_date = self.hparams["dataset"][0:8]

        self.inverse_y_transform = joblib.load(
            self.hparams['inverse_transform'])

        if self.hparams['hyper_search']:
            suggester_trial = trial
        else:
            suggester_trial = frozen_trial()

        # CMA-ES cannot work with categoricals, to ensure the use of CMA-ES for optimizer selection we must make it an int
        self.optuna_optimizer = "AdamW"
        self.optuna_weight_decay = suggester_trial.suggest_float(
            "weight_decay", 1e-3, 0.1, log=True)

        if self.optuna_optimizer == "SGD":
            self.optuna_momentum = suggester_trial.suggest_float("momentum",
                                                                 1e-2,
                                                                 0.9,
                                                                 log=True)
        else:
            self.optuna_momentum = 0.0

        # If network == -3, let optuna choose the network
        if self.hparams["network"] == -3:
            self.optuna_network = suggester_trial.suggest_categorical(
                "network", [1, 2, 3, 4])
            if self.optuna_network > 2:
                self.weather_data = suggester_trial.suggest_categorical(
                    "historical_weather", [1, 2])
        # Otherwise use the provided network
        else:
            self.optuna_network = self.hparams["network"]
            self.weather_data = self.hparams['historical_weather']

        # Learning rate
        if self.hparams['one_cycle'] == 1:
            self.lr = self.hparams['lr']
        else:
            self.lr = suggester_trial.suggest_float("learning_rate",
                                                    1e-8,
                                                    5e-2,
                                                    log=True)

        self.model, self.hyperparams, self.ntw = suggest_network(
            suggester_trial, self.hparams, self.optuna_network,
            self.batch_size, self.optuna_optimizer, self.lr,
            self.optuna_momentum, self.optuna_weight_decay)

        if self.hparams["cross_entropy"]:
            self.loss_fn = nn.CrossEntropyLoss()
            self.performance_measure = "acc"
        else:
            self.performance_measure = "R^2"
            self.loss_fn = nn.MSELoss()

        # Prepare data for run
        self.dataset, self.train_sampler, self.val_sampler, self.test_sampler, self.train_indices = self.prepare_data(
        )

    def prepare_data(self):

        self.data_path = self.data_dir

        add_to_path = ""

        if self.weather_data == 1:
            add_to_path += self.hparams["dataset"]

            # Create example input array for the batch_size finder to work
            self.example_input_array = [[{
                "input_genome":
                torch.randn(1, self.hparams['gene_length'],
                            self.hparams['gene_size']),
                "air_temp":
                torch.randn(1, self.hparams['weather_length']),
                "precip":
                torch.randn(1, self.hparams['weather_length']),
            }]]

        elif self.weather_data == 2:
            add_to_path += self.hparams["dataset"]

            # Create example input array for the batch_size finder to work
            self.example_input_array = [[{
                "input_genome":
                torch.randn(1, self.hparams['gene_length'],
                            self.hparams['gene_size']),
                "air_temp":
                torch.randn(1, self.hparams['weather_length']),
                "precip":
                torch.randn(1, self.hparams['weather_length']),
            }]]
        else:
            add_to_path += self.hparams["dataset"]

            # Create example input array for the batch_size finder to work
            self.example_input_array = [[{
                "input_genome":
                torch.randn(1, self.hparams['gene_length'],
                            self.hparams['gene_size']),
                "input_weather":
                torch.randn(1, 2),
            }]]
        if self.hparams['historical_weather']:
            add_to_path = add_to_path.replace('mean', 'historical')

        self.data_path += add_to_path

        # Enter a set of user attrs in the optuna database for ease of use
        self.trial.set_user_attr("dataset", self.data_path)
        self.trial.set_user_attr("using_onecycle_schedule",
                                 self.hparams.one_cycle)
        self.trial.set_user_attr("batch_size", self.batch_size)
        self.trial.set_user_attr("data_path", self.data_path)
        self.trial.set_user_attr("learning_rate", self.lr)
        self.trial.set_user_attr("sampler", self.hparams.sampler)
        self.trial.set_user_attr("nystrom_attention",
                                 self.hparams.nystrom_attention)
        self.trial.set_user_attr("augment_data_using_input_dropouts",
                                 self.hparams.dropout_input)
        if self.hparams.dropout_input:
            self.trial.set_user_attr("dropout_input_rate",
                                     self.hparams.dropout_input_rate)
        self.trial.set_user_attr("git_commit_hash", get_git_revision_hash())

        # Print the data path we're using
        print(self.data_path)
        dataset = pxt.TorchDataset(self.data_path)

        # Creating data indices for our entire dataset
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        if self.hparams.hyper_search:
            # Use all data as hyperparameter search data
            train = indices
        else:
            # Split the dataset into 10 folds using a fixed seed
            print("Fetching fold: " + str(self.trial.number))

            # Indices were created in R which is not 0-indexed
            # We therefore need to subtract one from the indices to make them fit python
            # Also the names of the files are not 0-indexed
            test_indices = pd.read_csv("test" + str(self.trial.number + 1) +
                                       ".csv").values.squeeze() - 1

            # We construct our training data by deleting the test indices from the total dataset indices
            train = np.delete(arr=indices, obj=test_indices)
            test_sampler = SubsetRandomSampler(test_indices)

        # We split the training data into training and validation data
        split = int(np.floor(0.20 * len(train)))
        train_indices, val_indices = train[split:], train[:split]

        # We then create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        if self.hparams.hyper_search:
            return dataset, train_sampler, val_sampler, None, train_indices
        else:
            return dataset, train_sampler, val_sampler, test_sampler, train_indices

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        y = batch["target"]

        logits = self([batch])

        if self.hparams["cross_entropy"]:
            loss = self.loss_fn(logits, y.long())
            _, predicted = torch.max(logits.data, 1)

        else:
            # Scale logits and targets to get values over 1
            logits = logits
            targets = y[:, 0]
            loss = self.loss_fn(logits.squeeze(), targets.float())

        self.log("train_loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        y = batch["target"]

        logits = self([batch])

        if self.hparams["cross_entropy"]:
            loss = self.loss_fn(logits, y.long())
            _, predicted = torch.max(logits.data, 1)

            l1 = np.inf
        else:
            # Scale logits and targets to get values over 1
            logits = logits
            targets = y[:, 0]
            loss = self.loss_fn(logits.squeeze(), targets.float())

            try:
                l1 = self.l1(
                    self.inverse_y_transform.inverse_transform(
                        logits.squeeze().detach().cpu().numpy().reshape(-1,
                                                                        1)),
                    self.inverse_y_transform.inverse_transform(
                        targets.detach().cpu().numpy().reshape(-1, 1)))
            except ValueError:
                l1 = np.inf

        output = OrderedDict({
            "loss": loss,
            "l1_distance": l1,
        })
        return output

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_l1_mean = 0
        for output in outputs:
            val_loss_mean += output["loss"]
            val_l1_mean += output["l1_distance"]
        val_loss_mean = val_loss_mean / len(outputs)
        val_l1_mean = val_l1_mean / len(outputs)

        self.log("val_loss", val_loss_mean, on_epoch=True, prog_bar=True)
        self.log(
            "val_l1",
            val_l1_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # Log mean l1 distance
        self.trial.set_user_attr("Mean_l1_distance", val_l1_mean)

        # Only log hyperparams when the loss strictly improves and we are not doing sanity checks
        if self.batch_nb > 1 and val_loss_mean < self.best_loss:
            self.best_loss = val_loss_mean
            self.log_validation(self.logger.experiment, {
                'val_l1_log': val_l1_mean,
                'val_loss_log': val_loss_mean
            })

    def log_validation(self, logging, metrics):
        # Log different things depending on the network we are using
        if self.optuna_network == 1:
            run_name = 'CNN'
        elif self.optuna_network == 2:
            run_name = 'ResNet'
        elif self.optuna_network == 3:
            run_name = 'Performer'
        elif self.optuna_network == 4:
            run_name = 'Historical_Performer'
        elif self.optuna_network == 5:
            run_name = 'Multimodal_Performer'

        logging.add_hparams(self.hyperparams, metrics, run_name=run_name)

    def test_step(self, batch, batch_nb):
        y = batch['target']
        x = batch['input_genome']
        location = batch['location']

        logits = self([batch])

        for i in range(location.shape[0]):
            self.locations.append(location[i].detach().cpu().numpy())
            self.ys.append(y[i, 0].detach().cpu().numpy())
            self.preds.append(logits[i, 0].detach().cpu().numpy())
            self.xs.append(x[i].detach().cpu().numpy())

        if self.hparams["cross_entropy"]:
            loss = self.loss_fn(logits, y.long())
            _, predicted = torch.max(logits.data, 1)

            l1 = np.inf
        else:
            # Scale logits and targets to get values over 1
            logits = logits
            targets = y[:, 0]
            loss = self.loss_fn(logits.squeeze(), targets.float())

            try:
                l1 = self.l1(
                    self.inverse_y_transform.inverse_transform(
                        logits.squeeze().detach().cpu().numpy().reshape(-1,
                                                                        1)),
                    self.inverse_y_transform.inverse_transform(
                        targets.detach().cpu().numpy().reshape(-1, 1)))
            except ValueError:
                l1 = np.inf

        output = OrderedDict({
            "loss": loss,
            "l1_distance": l1,
        })
        return output

    def test_epoch_end(self, outputs):
        test_loss_mean = 0
        test_l1_mean = 0
        for output in outputs:
            test_loss_mean += output["loss"]
            test_l1_mean += output["l1_distance"]
        test_loss_mean = test_loss_mean / len(outputs)
        test_l1_mean = test_l1_mean / len(outputs)

        self.log("test_loss", test_loss_mean, on_epoch=True, prog_bar=True)
        self.log(
            "test_l1_mean",
            test_l1_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        now = datetime.datetime.now().strftime("%d%m%Y-%H:%M:%S")
        locations_df = pd.DataFrame(self.locations)
        y_df = pd.DataFrame(self.ys)
        x_np = np.asarray(self.xs)
        pred_df = pd.DataFrame(self.preds)

        locations_df.to_csv("pandas/" + self.ntw + '/locations_' +
                            str(self.trial.number) + '_' + now + '.csv')

        y_df.to_csv("pandas/" + self.ntw + '/ys_' + str(self.trial.number) +
                    '_' + now + '.csv')

        pred_df.to_csv("pandas/" + self.ntw + '/preds_' +
                       str(self.trial.number) + '_' + now + '.csv')
        np.save(
            "pandas/" + self.ntw + '/xs_' + str(self.trial.number) + '_' +
            now + '.npy', x_np)

    def configure_optimizers(self):
        num_steps_per_epoch = max(
            len(self.train_indices) // self.batch_size, 1) + 1
        total_steps = num_steps_per_epoch * (self.hparams["num_epochs"] -
                                             1) + 1

        print('num train examples: ' + str(len(self.train_indices)))
        print('num steps per epoch: ' + str(num_steps_per_epoch))
        print('batch size: ' + str(self.batch_size))
        print('num epochs: ' + str(self.hparams["num_epochs"]))
        print('total steps: ' + str(total_steps))

        if self.optuna_optimizer == "AdamW":
            self.optimizer = AdamW(params=self.parameters(),
                                   lr=self.lr,
                                   amsgrad=False)

        elif self.optuna_optimizer == "SGD":
            self.optimizer = SGD(params=self.parameters(), lr=self.lr)

        elif self.optuna_optimizer == "RProp":
            self.optimizer = Rprop(params=self.parameters(), lr=self.lr)

        print(self.optimizer)
        self.hyperparams['batch_size'] = self.batch_size
        if self.hparams.one_cycle:
            self.scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.hparams['lr'],
                cycle_momentum=False,
                # epochs=self.hparams["num_epochs"],
                # steps_per_epoch=num_steps_per_epoch,
                total_steps=total_steps,
            )

            sched = {
                'scheduler': self.scheduler,
                'name': 'one_cycle_scheduler',
                'interval': 'step'
            }

            self.hyperparams['lr'] = self.hparams['lr']
            print('lr: ' + str(self.hparams['lr']))
            print('------------------------------------------')

            return [self.optimizer], [sched]
        print('lr: ' + str(self.lr))
        print('------------------------------------------')
        return self.optimizer

    def train_dataloader(self):
        use_cuda = True and torch.cuda.is_available()
        kwargs = {
            "num_workers": 4,
            "pin_memory": True
        } if use_cuda else {
            "num_workers": 4
        }
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           sampler=self.train_sampler,
                                           **kwargs)

    def val_dataloader(self):
        use_cuda = True and torch.cuda.is_available()
        kwargs = {
            "num_workers": 4,
            "pin_memory": True
        } if use_cuda else {
            "num_workers": 4
        }
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           sampler=self.val_sampler,
                                           **kwargs)

    def test_dataloader(self):
        use_cuda = True and torch.cuda.is_available()
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        return torch.utils.data.DataLoader(self.dataset,
                                           batch_size=2,
                                           sampler=self.test_sampler,
                                           **kwargs)
