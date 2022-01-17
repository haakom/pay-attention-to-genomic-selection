from test_tube.hpc import SlurmCluster


def optimize_on_cluster(
    hyperparams,
    slurm_log_dir: str = "Slurm",
    nb_nodes: int = 10,
    trial_code=None,
    job_name: str = "--insert-job-name-here",
    num_trials_per_node: int = 10,
):

    # init cluster
    cluster = SlurmCluster(hyperparam_optimizer=hyperparams,
                           log_path=slurm_log_dir)

    # let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
    cluster.notify_job_status(email="example@yourdomain.com",
                              on_done=False,
                              on_fail=False)

    # set the job options. In this instance, we'll run nb_trials different models
    # each with its own set of hyperparameters giving each one 1 GPU (ie: taking up nb_trials GPUs)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1

    # set slurm partition and account
    cluster.add_slurm_cmd(cmd="partition",
                          value="your_partition_here",
                          comment="what partition to use")

    cluster.add_slurm_cmd(cmd="account",
                          value="your_account_here",
                          comment="what account to use")

    # we'll request 10GB of memory per node
    cluster.memory_mb_per_node = 10000

    # set a walltime
    cluster.job_time = "1-20:00:00"

    # Set up environment
    job_name = job_name
    job_disp_name = job_name

    # Run code
    cluster.optimize_parallel_cluster_gpu(
        trial_code,
        nb_trials=nb_nodes,
        job_name=job_name,
        job_display_name=job_disp_name,
    )
