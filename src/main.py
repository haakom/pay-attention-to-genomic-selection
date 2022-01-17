import os

import numpy as np
from test_tube.hpc import HyperOptArgumentParser

from optuna_search import run_trials
from slurm_single_gpu import optimize_on_cluster

if __name__ == "__main__":
    num_trials = 10  #10-fold CV
    num_nodes_to_use = 5
    num_trials_per_node = num_trials // num_nodes_to_use
    parser = HyperOptArgumentParser(strategy="grid_search", add_help=True)

    ##########################  Experiment params #################################
    parser.add_argument(
        "--hyper_search",
        type=int,
        default=1,
        help=
        "If we are doing a hyperparameter search (1) or crossvalidation run, default = 1 = hyperparameter search",
    )
    parser.add_argument(
        "--total_num_trials",
        type=int,
        default=num_trials,
        help="How many trials to run in the study, default: 400",
    )
    parser.add_argument(
        "--database_name",
        type=str,
        default="/path/to/hyperparameter_search_database",
        help="Name of the database from which to load the frozen trial",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="/path/to/datasets",
        help="Where the datasets are stored",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/dataset/name",
        help="The name of dataset to use",
    )
    parser.add_argument(
        "--inverse_transform",
        type=str,
        default="/path/to/inverse_transform",
        help="The path to the inverse transform of the target data",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed used to seed training",
    )
    parser.add_argument(
        "--validation_and_test_split",
        type=float,
        default=0.2,
        help="fraction of data used for validation",
    )
    parser.add_argument(
        "--standardized_weather_data",
        type=int,
        default=2,
        help=
        "whether to standardize, zeroone or leave as is the weather data, default = zoeroone",
    )
    parser.add_argument(
        "--standardize_genome",
        type=int,
        default=1,
        help="whether to standardize dataset genome, default = True",
    )
    parser.add_argument(
        "--historical_weather",
        type=int,
        default=0,
        help=
        "whether to use historical weather data or one mean value, default = mean vaule (0)",
    )
    parser.add_argument(
        "--cross_entropy",
        type=int,
        default=0,
        help=
        "if we are running classification using cross entropy or regression, default = regression",
    )
    parser.add_argument(
        "--one_cycle",
        type=int,
        default=0,
        help="whether to use onecycle learning rate, default = 0 = no",
    )
    parser.add_argument(
        "--sampler",
        type=int,
        default=1,
        help=
        "What sampler we are using in hyperparameter search: 0 = TPE, 1 = CMA-ES+TPE, default = 1",
    )
    parser.add_argument(
        "--num_hparams_explor",
        type=int,
        default=100,
        help="How many trials to use to for exploration, default = 100",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=int,
        default=1,
        help="whether to shuffle dataset or not",
    )
    parser.add_argument(
        "--separate_embedding",
        type=int,
        default=0,
        help=
        "whether to use separate positional encodings for gene and weather data , default = 0 = False",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--find_batch_size",
        type=int,
        default=1,
        help=
        "Should we try to find optimal batch size or not? Default = 1 = Yes we should",
    )
    parser.add_argument(
        "--pruning",
        default=1,
        help="Whether to prune unpromising trials. Default = True.",
    )
    parser.add_argument(
        "--gene_length",
        type=int,
        default=13321,
        help="Size of gene we're using. Default = 13321, alt = 10679",
    )
    parser.add_argument(
        "--weather_length",
        type=int,
        default=123,
        help="Size of gene we're using. Default = 200",
    )
    parser.add_argument(
        "--gene_size",
        type=int,
        default=12,
        help="Size of gene we're using. Default = 12",
    )
    ####################### Hardware Params ######################################
    parser.add_argument(
        "--slurm",
        type=int,
        default=0,
        help=
        "Wether to run on slurm or local single machine, default = 0 = local machine",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=num_nodes_to_use,
        help="How many nodes to use, default: 10",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=1,
        help="whether to use gpus",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="how many gpus to use",
    )
    parser.add_argument(
        "--num_trials_per_node",
        type=int,
        default=num_trials_per_node,
        help=
        "How many trials to run per node, default: num_trails // num_nodes",
    )
    parser.add_argument(
        "--dropout_input",
        type=int,
        default=0,
        help=
        "Should we do augmentation of input through dropouts? default = no = 0",
    )
    parser.add_argument(
        "--dropout_input_rate",
        type=float,
        default=0.3,
        help="Dropout rate at input augmentation, default = 0.3",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Set a learning rate (overwritten), default = 1e-3",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=128,
        help="how many epochs to train for",
    )
    parser.add_argument(
        "--show_lr",
        type=int,
        default=0,
        help="Whether to show the onecycle learning rate, default = 0 = no",
    )

    ############################## Model args #################################
    parser.add_argument(
        "--network",
        type=int,
        default=1,
        help=
        "whether to use a Vanilla CNN: 1, ResNet: 2, Performer: 3, Historical Performer: 4, Multimodal Performer: 5 , default = Vanilla CNN: 1",
    )
    parser.add_argument(
        "--nystrom_attention",
        type=int,
        default=0,
        help="whether to use a Nyström attention or not , default = 0 = False",
    )

    ############################ Set up hyperparameter search on cluster #########
    parser.opt_list(
        "--run_num",
        type=int,
        default=0,
        options=np.arange(num_trials),
        tunable=True,
    )

    ########################  Set up experiment and run ##############################
    fileDir = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(fileDir)
    parentDir = os.path.dirname(parentDir)
    save_model_dir = os.path.join(parentDir, "Results/")
    slurm_log_dir = os.path.join(parentDir, "Slurm")

    parser.add_argument(
        "--logdir",
        type=str,
        default=save_model_dir,
        help="Where logs should be saved, default is ../../Results/")

    MODEL_DIR = os.path.join(save_model_dir, "optuna")

    parser.add_argument(
        "--model_dir",
        type=str,
        default=MODEL_DIR,
        help="Where the model should be saved, default is ../../Results/optuna"
    )

    hyperparser = parser.parse_args()

    if hyperparser.slurm:
        optimize_on_cluster(
            hyperparams=hyperparser,
            slurm_log_dir=slurm_log_dir,
            nb_nodes=num_nodes_to_use,
            job_name="X-VAL",
            trial_code=run_trials,
            num_trials_per_node=num_trials_per_node,
        )
    else:
        run_trials(ttargs=hyperparser, )
